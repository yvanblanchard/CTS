# cts_geometry.py
import numpy as np
from scipy.optimize import fsolve

class Surface:
    """Class representing a doubly curved surface S(u,v) as described in the paper"""
    def __init__(self):
        # Surface parameters based on dimensions from paper Fig. 6
        self.length = 544.0  # mm (X dimension)
        self.width = 394.0   # mm (Y dimension)
        self.height = 110.0  # mm (Z dimension)
        
        # Extended range for smooth edges
        self.u_range = [-self.width/2 * 1.1, self.width/2 * 1.1]
        self.v_range = [-self.length/2 * 1.1, self.length/2 * 1.1]
        
        # Surface shape parameters
        self.A = 0.5  # amplitude coefficient
        self.B = 0.3  # twist coefficient
        self.base_height = 0.2  # relative base height
    
    def evaluate(self, u, v):
        """Evaluate surface point at parameters (u,v)"""
        # Normalize parameters to [-1, 1]
        u_norm = u / (self.width/2)
        v_norm = v / (self.length/2)
        
        # Calculate saddle shape with twist
        z_saddle = self.height * self.A * (u_norm**2 - v_norm**2)
        z_twist = self.height * self.B * (u_norm * v_norm)
        
        # Apply edge smoothing
        edge_smooth = (1 - u_norm**4) * (1 - v_norm**4)
        z = (z_saddle + z_twist) * edge_smooth
        z += self.height * self.base_height
        
        return np.array([u, v, z])
    
    def normal(self, u, v):
        """Calculate surface normal at point (u,v)"""
        delta = 1e-7
        # Calculate partial derivatives
        du = (self.evaluate(u + delta, v) - self.evaluate(u - delta, v)) / (2 * delta)
        dv = (self.evaluate(u, v + delta) - self.evaluate(u, v - delta)) / (2 * delta)
        
        # Normal is cross product of tangent vectors
        normal = np.cross(du, dv)
        return normal / np.linalg.norm(normal)

class SphereIntersection:
    """Implementation of sphere intersection technique from Section 2.3"""
    def __init__(self, surface):
        self.surface = surface
    
    def find_intersection_circle(self, P1, Q1, w, d):
        """
        Calculate intersection circle between spheres.
        P1: Center of width sphere (green)
        Q1: Center of segment length sphere (red)
        w: Width sphere radius (tow width)
        d: Distance sphere radius (segment length)
        """
        # Vector from P1 to Q1
        v = Q1 - P1
        v_norm = np.linalg.norm(v)
        
        # Check intersection conditions
        if v_norm > (w + d) or v_norm < abs(w - d) or v_norm == 0:
            print(f"No intersection possible: distance={v_norm:.2f}, w+d={w+d:.2f}, |w-d|={abs(w-d):.2f}")
            return None, None, None
            
        # Calculate intersection parameters
        h = (w**2 - d**2 + v_norm**2) / (2 * v_norm)  # equation (2)
        r0 = np.sqrt(w**2 - h**2)  # equation (4)
        
        # Circle center
        v_unit = v / v_norm
        C0 = P1 + h * v_unit  # equation (5)
        
        return C0, r0, v

class PinJointedStrip:
    """Implementation of pin-jointed strip model for CTS"""
    def __init__(self, surface, tow_width, segment_length):
        self.surface = surface
        self.w = tow_width
        self.d = segment_length
        self.sphere_intersector = SphereIntersection(surface)
    
    
    def find_Q_point(self, P_current, P_next, Q_prev):
        """
        Find next Q point using sphere intersection method.
        Selects intersection point based on vector alignment with reference path.
        
        Args:
            P_current: Current reference path point (for direction)
            P_next: Next reference path point
            Q_prev: Previous shifted point
        """
        # Get intersection circle parameters
        C0, r0, v = self.sphere_intersector.find_intersection_circle(P_next, Q_prev, self.w, self.d)
        
        if C0 is None:
            print("No intersection circle found")
            return None
            
        # Calculate reference direction vector
        ref_direction = P_next - P_current
        ref_direction = ref_direction / np.linalg.norm(ref_direction)
        
        # Create basis for circle plane
        v_unit = v / np.linalg.norm(v)
        if abs(v_unit[2]) < 0.9:
            temp = np.array([0, 0, 1])
        else:
            temp = np.array([1, 0, 0])
        
        u1 = np.cross(v_unit, temp)
        u1 = u1 / np.linalg.norm(u1)
        u2 = np.cross(v_unit, u1)
        
        def surface_intersection(theta):
            """Return single value for intersection error"""
            point = C0 + r0 * (u1 * np.cos(theta[0]) + u2 * np.sin(theta[0]))
            surface_point = self.surface.evaluate(point[0], point[1])
            return np.linalg.norm(surface_point - point)
        
        # Try both possible intersection points
        solutions = []
        
        # Try angles in two opposite regions of the circle
        for base_theta in [0, np.pi]:  # Try both halves of the circle
            for delta in [-0.1, 0, 0.1]:  # Try slight variations around each base angle
                theta_init = base_theta + delta
                try:
                    sol = fsolve(surface_intersection, [theta_init], full_output=True)
                    if sol[2] == 1:  # Check convergence
                        theta = sol[0][0]
                        point = C0 + r0 * (u1 * np.cos(theta) + u2 * np.sin(theta))
                        Q = self.surface.evaluate(point[0], point[1])
                        error = surface_intersection([theta])
                        
                        if error < 1e-3:  # Valid surface intersection
                            solutions.append(Q)
                            
                except Exception as e:
                    continue
        
        # Find solution with positive scalar product
        best_Q = None
        best_alignment = float('-inf')
        
        for Q in solutions:
            # Calculate Q direction vector
            Q_direction = Q - Q_prev
            Q_direction = Q_direction / np.linalg.norm(Q_direction)
            
            # Calculate scalar product with reference direction
            alignment = np.dot(Q_direction, ref_direction)
            
            if alignment > best_alignment:
                best_alignment = alignment
                best_Q = Q
        
        if best_Q is not None:
            # Verify geometric constraints
            dist_to_P = np.linalg.norm(best_Q - P_next)
            dist_to_Q_prev = np.linalg.norm(best_Q - Q_prev)
            Q_direction = best_Q - Q_prev
            Q_direction = Q_direction / np.linalg.norm(Q_direction)
            final_alignment = np.dot(Q_direction, ref_direction)
            
            print(f"Selected Q point analysis:")
            print(f"Distance to P_next: {dist_to_P:.2f} mm (should be {self.w:.2f})")
            print(f"Distance to Q_prev: {dist_to_Q_prev:.2f} mm (should be {self.d:.2f})")
            print(f"Vector alignment: {final_alignment:.3f}")
            return best_Q
        
        print("Could not find satisfactory solution.")
        return None

    def propagate_path(self, reference_path, initial_shift_point):
        """Propagate path using vector alignment for point selection"""
        Q_points = [initial_shift_point]  # Q0
        
        for i in range(1, len(reference_path)):
            P_current = reference_path[i-1]  # Previous P point
            P_next = reference_path[i]     # Current P point
            Q_prev = Q_points[-1]          # Previous Q point
            
            print(f"\nFinding Q{i}:")
            Q_next = self.find_Q_point(P_current, P_next, Q_prev)
            if Q_next is None:
                print(f"Could not find Q{i}")
                break
                
            Q_points.append(Q_next)
        
        return Q_points


def calculate_curvature(points, surface):
    """
    Calculate curvature radius at points using surface derivatives.
    Returns array of curvature radii at each point.
    """
    radii = []
    
    for point in points:
        # Get surface derivatives at point
        u, v = point[0], point[1]
        
        # Calculate first derivatives
        h = 1e-6  # Small step for numerical derivatives
        
        # du/dx derivatives
        pt_du = surface.evaluate(u + h, v)
        pt_du_neg = surface.evaluate(u - h, v)
        du = (pt_du - pt_du_neg) / (2*h)
        
        # dv/dx derivatives
        pt_dv = surface.evaluate(u, v + h)
        pt_dv_neg = surface.evaluate(u, v - h)
        dv = (pt_dv - pt_dv_neg) / (2*h)
        
        # Second derivatives
        du2 = (pt_du + pt_du_neg - 2*point) / h**2
        dv2 = (pt_dv + pt_dv_neg - 2*point) / h**2
        
        # Cross derivative
        pt_duv = surface.evaluate(u + h, v + h)
        pt_du_neg_v = surface.evaluate(u - h, v + h)
        pt_du_v_neg = surface.evaluate(u + h, v - h)
        pt_du_neg_v_neg = surface.evaluate(u - h, v - h)
        duv = (pt_duv + pt_du_neg_v_neg - pt_du_neg_v - pt_du_v_neg) / (4*h**2)
        
        # Compute first fundamental form coefficients
        E = np.dot(du, du)
        F = np.dot(du, dv)
        G = np.dot(dv, dv)
        
        # Compute second fundamental form coefficients
        n = np.cross(du, dv)
        n = n / np.linalg.norm(n)
        
        L = np.dot(du2, n)
        M = np.dot(duv, n)
        N = np.dot(dv2, n)
        
        # Compute Gaussian curvature
        K = (L*N - M**2) / (E*G - F**2)
        
        # Compute mean curvature
        H = (E*N - 2*F*M + G*L) / (2*(E*G - F**2))
        
        # Compute principal curvatures
        k1 = H + np.sqrt(H**2 - K)
        k2 = H - np.sqrt(H**2 - K)
        
        # Maximum curvature radius is inverse of minimum absolute curvature
        k_max = max(abs(k1), abs(k2))
        if k_max < 1e-10:  # Almost flat
            radius = float('inf')
        else:
            radius = 1.0 / k_max
        
        radii.append(radius)
    
    return np.array(radii)

def calculate_curvature_from_points(points):
    """Calculate curvature using three consecutive points method"""
    radii = np.zeros(len(points))
    points = np.array(points)
    
    for i in range(1, len(points)-1):
        p1 = points[i-1]
        p2 = points[i]
        p3 = points[i+1]
        
        # Calculate vectors and distances
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p3 - p1)
        
        # Calculate semi-perimeter
        s = (a + b + c) / 2.0
        
        # Calculate triangle area using Heron's formula
        area = np.sqrt(s * (s-a) * (s-b) * (s-c))
        
        # Calculate radius using R = abc/(4A)
        if area > 1e-10:  # Avoid division by zero
            radii[i] = (a * b * c) / (4.0 * area)
        else:
            radii[i] = float('inf')
    
    # Handle endpoints
    radii[0] = radii[1]
    radii[-1] = radii[-2]
    
    return radii

def analyze_path_curvature(points):
    """Analyze and print curvature values"""
    radii = calculate_curvature_from_points(points)
    
    print("\nCurvature Analysis:")
    for i, radius in enumerate(radii):
        if radius > 1000:
            print(f"Point P{i}: >1000 mm")
        else:
            print(f"Point P{i}: {radius:.1f} mm")
    
    finite_radii = radii[radii < 1000]
    if len(finite_radii) > 0:
        print(f"\nMinimum radius: {np.min(finite_radii):.1f} mm")
        print(f"Maximum radius: {np.max(finite_radii):.1f} mm")


def create_ruled_surfaces(P_points, TCP_points, Q_points, num_points=10):
    """Create mesh points for ruled surfaces along entire path"""
    t = np.linspace(0, 1, num_points)
    all_quads_p_tcp = []
    all_quads_tcp_q = []
    
    for i in range(len(P_points)-1):
        # Get points for current segment
        p0, p1 = P_points[i], P_points[i+1]
        tcp0, tcp1 = TCP_points[i], TCP_points[i+1]
        q0, q1 = Q_points[i], Q_points[i+1]
        
        # P-TCP quadrilateral
        quad_points = []
        for j in range(num_points):
            p_line = p0 + t[j] * (p1 - p0)
            tcp_line = tcp0 + t[j] * (tcp1 - tcp0)
            
            for k in range(num_points):
                point = p_line + t[k] * (tcp_line - p_line)
                quad_points.append(point)
        all_quads_p_tcp.append(np.array(quad_points))
        
        # TCP-Q quadrilateral
        quad_points = []
        for j in range(num_points):
            tcp_line = tcp0 + t[j] * (tcp1 - tcp0)
            q_line = q0 + t[j] * (q1 - q0)
            
            for k in range(num_points):
                point = tcp_line + t[k] * (q_line - tcp_line)
                quad_points.append(point)
        all_quads_tcp_q.append(np.array(quad_points))
    
    return all_quads_p_tcp, all_quads_tcp_q
