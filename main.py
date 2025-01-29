# main.py with enhanced debugging for initial setup
import numpy as np
import plotly.graph_objects as go
from cts_geometry import Surface, SphereIntersection, PinJointedStrip, calculate_curvature, calculate_curvature_from_points, analyze_path_curvature, create_ruled_surfaces
from scipy.optimize import fsolve

def create_discrete_colormap(min_radius, max_radius, num_colors=10):
    """Create discrete colormap for radius visualization"""
    # Create logarithmically spaced boundaries
    boundaries = np.logspace(np.log10(min_radius), np.log10(max_radius), num_colors + 1)
    
    # Create colors from red to green
    colors = [
        '#00FF00',  # Green (large radius)
        '#33FF00',
        '#66FF00',
        '#99FF00',
        '#CCFF00',
        '#FFFF00',  # Yellow
        '#FFCC00',
        '#FF9900',
        '#FF6600',
        '#FF0000'   # Red (small radius)
    ]
    
    return boundaries, colors[::-1]  # Reverse colors for correct mapping

class CTSAnalyzer:
    def __init__(self, surface):
        self.surface = surface
        self.sphere_intersector = SphereIntersection(surface)
        
    def calculate_tcp_frame(self, P, Q, surface):
        """
        Calculate TCP point and its coordinate frame.
        
        Args:
            P: Reference path point
            Q: Shifted path point
            surface: Surface object
            
        Returns:
            TCP: TCP origin point
            frame: 3x3 array [X,Y,Z] of frame vectors
        """
        # Calculate TCP origin as midpoint
        TCP = 0.5 * (P + Q)
        
        # Project TCP onto surface
        TCP = surface.evaluate(TCP[0], TCP[1])
        
        # Get surface normal at TCP (Z axis)
        Z = surface.normal(TCP[0], TCP[1])
        Z = Z / np.linalg.norm(Z)
        
        # Y axis is along tow direction (P to Q)
        Y = Q - P
        Y = Y / np.linalg.norm(Y)
        
        # X axis is cross product to ensure orthonormal frame
        X = np.cross(Y, Z)
        X = X / np.linalg.norm(X)
        
        # Recalculate Y to ensure perfect orthogonality
        Y = np.cross(Z, X)
        
        return TCP, np.array([X, Y, Z])
    
    def add_tcp_frames(self, fig, P0, P1, Q0, Q1):
        """Add TCP points and their coordinate frames to the visualization"""
        # Calculate TCP frames
        TCP1, frame1 = self.calculate_tcp_frame(P0, Q0, self.surface)
        TCP2, frame2 = self.calculate_tcp_frame(P1, Q1, self.surface)
        
        # Scale for frame axes visualization
        scale = self.surface.tow_width * 0.3
        
        # Add TCP points
        for i, (TCP, label) in enumerate([(TCP1, 'TCP1'), (TCP2, 'TCP2')]):
            fig.add_trace(go.Scatter3d(
                x=[TCP[0]],
                y=[TCP[1]],
                z=[TCP[2]],
                mode='markers+text',
                marker=dict(size=8, color='purple'),
                text=[label],
                textposition="top center",
                name=label
            ))
        
        # Add coordinate frames
        colors = ['red', 'green', 'blue']  # X, Y, Z axes
        names = ['X', 'Y', 'Z']
        
        for TCP, frame, label in [(TCP1, frame1, 'TCP1'), (TCP2, frame2, 'TCP2')]:
            for i, (axis, color, name) in enumerate(zip(frame, colors, names)):
                end_point = TCP + axis * scale
                fig.add_trace(go.Scatter3d(
                    x=[TCP[0], end_point[0]],
                    y=[TCP[1], end_point[1]],
                    z=[TCP[2], end_point[2]],
                    mode='lines',
                    line=dict(color=color, width=3),
                    name=f'{label} {name}-axis' if label == 'TCP1' else None,
                    showlegend=False #(label == 'TCP1')  # Show legend only for first TCP
                ))
    
    def create_surface_mesh(self, nu=100, nv=100):
        """Create surface mesh points for visualization"""
        u = np.linspace(self.surface.u_range[0], self.surface.u_range[1], nu)
        v = np.linspace(self.surface.v_range[0], self.surface.v_range[1], nv)
        U, V = np.meshgrid(u, v)
        
        X = np.zeros_like(U)
        Y = np.zeros_like(U)
        Z = np.zeros_like(U)
        
        for i in range(nu):
            for j in range(nv):
                point = self.surface.evaluate(U[i,j], V[i,j])
                X[i,j] = point[0]
                Y[i,j] = point[1]
                Z[i,j] = point[2]
                
        return X, Y, Z
    
    
    
    def get_discrete_color_scale(self, min_radius, max_radius):
        """
        Create a discrete color scale with 10 levels.
        Returns color boundaries and corresponding RGB colors.
        """
        # Create 10 logarithmically spaced boundaries
        boundaries = np.logspace(np.log10(min_radius), np.log10(max_radius), 11)
        
        # Create 10 colors from red to green
        colors = [
            '#FF0000',  # Red
            '#FF3300',
            '#FF6600',
            '#FF9900',
            '#FFCC00',
            '#FFFF00',  # Yellow
            '#CCFF00',
            '#99FF00',
            '#66FF00',
            '#00FF00'   # Green
        ]
        
        return boundaries, colors

    def calculate_tcp_frame(self, P, Q, surface):
        """
        Calculate TCP point and its coordinate frame.
        X axis is now inverted relative to the Y-Z cross product.
        """
        # Calculate TCP origin as midpoint
        TCP = 0.5 * (P + Q)
        TCP = surface.evaluate(TCP[0], TCP[1])
        
        # Get surface normal at TCP (Z axis)
        Z = surface.normal(TCP[0], TCP[1])
        Z = Z / np.linalg.norm(Z)
        
        # Y axis is along tow direction (P to Q)
        Y = Q - P
        Y = Y / np.linalg.norm(Y)
        
        # X axis is negative cross product to invert direction
        X = -np.cross(Y, Z)  # Negative to invert direction
        X = X / np.linalg.norm(X)
        
        # Recalculate Y to ensure perfect orthogonality
        Y = np.cross(Z, X)
        
        return TCP, np.array([X, Y, Z])

    def add_tcp_frames(self, fig, P_points, Q_points, num_display_q=2):
        """
        Add TCP points and their coordinate frames to the visualization.
        Only displays specified number of Q points.
        """
        # Scale for frame axes visualization - increased size
        scale = self.surface.tow_width * 0.5  # Increased from 0.3
        
        # Calculate all TCP frames
        tcp_frames = []
        for P, Q in zip(P_points, Q_points):
            TCP, frame = self.calculate_tcp_frame(P, Q, self.surface)
            tcp_frames.append((TCP, frame))
        
        # Add TCP points
        for i, (TCP, _) in enumerate(tcp_frames):
            fig.add_trace(go.Scatter3d(
                x=[TCP[0]],
                y=[TCP[1]],
                z=[TCP[2]],
                mode='markers+text',
                marker=dict(size=8, color='purple'),
                text=[f'TCP{i+1}'],
                textposition="top center",
                name=f'TCP{i+1}'
            ))
        
        # Add coordinate frames
        colors = ['red', 'green', 'blue']  # X, Y, Z axes
        names = ['X', 'Y', 'Z']
        
        for i, (TCP, frame) in enumerate(tcp_frames):
            for axis_idx, (axis, color, name) in enumerate(zip(frame, colors, names)):
                end_point = TCP + axis * scale
                fig.add_trace(go.Scatter3d(
                    x=[TCP[0], end_point[0]],
                    y=[TCP[1], end_point[1]],
                    z=[TCP[2], end_point[2]],
                    mode='lines',
                    line=dict(color=color, width=3),
                    name=f'TCP{i+1} {name}-axis' if i == 0 else None,
                    showlegend=False #(i == 0)  # Show legend only for first TCP
                ))
        
        return tcp_frames
        
    def visualize_intersection(self, P_points, Q_points, C0, r0, v, reference_path, 
                             show_spheres=True, num_display_q=2):
        """
        Visualize intersection with multiple TCP frames.
        Only displays first num_display_q Q points.
        """
        fig = go.Figure()
        
        # Add surface
        X, Y, Z = self.create_surface_mesh()
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Blues',
            opacity=0.6,
            showscale=False,
            name='Manufacturing Surface'
        ))

        # Calculate curvature radii for reference path
        #radii = calculate_curvature(reference_path, self.surface)
        radii = calculate_curvature_from_points(reference_path)

        # Create discrete color mapping
        min_radius = max(10, np.min(radii[radii != float('inf')]))
        max_radius = min(1000, np.max(radii[radii != float('inf')]))
        boundaries, colors = create_discrete_colormap(min_radius, max_radius)
        
        # Add reference path with curvature coloring
        ref_path = np.array(reference_path)
        for i in range(len(ref_path)-1):
            radius = radii[i]
            if radius == float('inf'):
                radius = max_radius
                
            # Find color level
            color_idx = np.searchsorted(boundaries, radius) - 1
            color_idx = min(color_idx, len(colors)-1)
            color = colors[color_idx]
            
            fig.add_trace(go.Scatter3d(
                x=ref_path[i:i+2,0],
                y=ref_path[i:i+2,1],
                z=ref_path[i:i+2,2],
                mode='lines',
                line=dict(color=color, width=4),
                showlegend=False
            ))

        # Add reference path points
        fig.add_trace(go.Scatter3d(
            x=ref_path[:,0],
            y=ref_path[:,1],
            z=ref_path[:,2],
            mode='markers+text',
            marker=dict(size=6, color='red'),
            text=[f'P{i}' for i in range(len(ref_path))],
            textposition="top center",
            showlegend=False
        ))
        
        # Add Q points (only first num_display_q points)
        Q_array = np.array(Q_points[:num_display_q])
        fig.add_trace(go.Scatter3d(
            x=Q_array[:,0],
            y=Q_array[:,1],
            z=Q_array[:,2],
            mode='markers+text',
            marker=dict(size=6, color='blue'),
            text=[f'Q{i}' for i in range(len(Q_array))],
            textposition="top center",
            showlegend=False
        ))
        
        # Add TCP frames
        tcp_frames = self.add_tcp_frames(fig, P_points, Q_points, num_display_q)
        
        # Add intersection spheres only for first point if requested
        if show_spheres and len(P_points) > 1 and len(Q_points) > 1:
            P1, Q0 = P_points[1], Q_points[0]
            phi = np.linspace(0, np.pi, 20)
            theta = np.linspace(0, 2*np.pi, 20)
            phi, theta = np.meshgrid(phi, theta)
            
            # Width sphere at P1
            x_w = P1[0] + self.surface.tow_width * np.sin(phi) * np.cos(theta)
            y_w = P1[1] + self.surface.tow_width * np.sin(phi) * np.sin(theta)
            z_w = P1[2] + self.surface.tow_width * np.cos(phi)
            
            fig.add_trace(go.Surface(
                x=x_w, y=y_w, z=z_w,
                colorscale=[[0, 'green'], [1, 'green']],
                opacity=0.3,
                showscale=False,
                name='Width Sphere'
            ))
            
            # Distance sphere at Q0
            x_d = Q0[0] + self.surface.segment_length * np.sin(phi) * np.cos(theta)
            y_d = Q0[1] + self.surface.segment_length * np.sin(phi) * np.sin(theta)
            z_d = Q0[2] + self.surface.segment_length * np.cos(phi)
            
            fig.add_trace(go.Surface(
                x=x_d, y=y_d, z=z_d,
                colorscale=[[0, 'red'], [1, 'red']],
                opacity=0.3,
                showscale=False,
                name='Distance Sphere'
            ))

        # Calculate all TCP points
        TCP_points = []
        for i in range(len(reference_path)):
            tcp = 0.5 * (reference_path[i] + Q_points[i])
            TCP_points.append(tcp)
        
        # Create ruled surfaces
        quads_p_tcp, quads_tcp_q = create_ruled_surfaces(
            reference_path, 
            TCP_points, 
            Q_points
        )
        
        # Add P-TCP quadrilaterals
        for quad_points in quads_p_tcp:
            fig.add_trace(go.Mesh3d(
                x=quad_points[:,0],
                y=quad_points[:,1],
                z=quad_points[:,2],
                color='lightblue',
                opacity=0.5,
                showscale=False,
                showlegend=False
            ))
        
        # Add TCP-Q quadrilaterals
        for quad_points in quads_tcp_q:
            fig.add_trace(go.Mesh3d(
                x=quad_points[:,0],
                y=quad_points[:,1],
                z=quad_points[:,2],
                color='lightgreen',
                opacity=0.5,
                showscale=False,
                showlegend=False
            ))
        
        # Add colorbar for curvature radius
        fig.add_trace(go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],
            mode='markers',
            marker=dict(
                colorscale=list(zip(
                    np.linspace(0, 1, len(colors)),
                    colors
                )),
                showscale=True,
                cmin=np.log10(min_radius),
                cmax=np.log10(max_radius),
                colorbar=dict(
                    title='Curvature Radius (mm)',
                    titleside='right',
                    x=1.0,
                    y=0.5,
                    len=0.8,
                    tickvals=np.linspace(
                        np.log10(min_radius),
                        np.log10(max_radius),
                        10
                    ),
                    ticktext=[f'{boundaries[i]:.0f}' for i in range(len(boundaries)-1)],
                    tickmode='array'
                )
            ),
            showlegend=False
        ))

        # Update layout
        fig.update_layout(
            title='Continuous Tow Shearing - Path generation on 3D surface',
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Z (mm)',
                aspectmode='data'
            ),
            showlegend=False,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.85,
                bgcolor="white",
                bordercolor="Black",
                borderwidth=1
            ),
            margin=dict(r=200)
        )
        
        fig.show()
        
        return tcp_frames    
    
    def calculate_curvature(self, points):
        """
        Calculate curvature radius at each point of the path.
        Using numerical approximation with three consecutive points.
        
        Args:
            points: Array of path points
            
        Returns:
            radii: Array of curvature radii
        """
        num_points = len(points)
        radii = np.zeros(num_points)
        
        for i in range(1, num_points-1):
            # Get three consecutive points
            p1 = points[i-1]
            p2 = points[i]
            p3 = points[i+1]
            
            # Calculate vectors and their norms
            v1 = p2 - p1
            v2 = p3 - p2
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm == 0 or v2_norm == 0:
                radii[i] = float('inf')
                continue
            
            # Normalize vectors
            v1_unit = v1 / v1_norm
            v2_unit = v2 / v2_norm
            
            # Calculate angle between vectors
            cos_theta = np.dot(v1_unit, v2_unit)
            cos_theta = np.clip(cos_theta, -1, 1)  # Avoid numerical errors
            theta = np.arccos(cos_theta)
            
            if theta == 0:
                radii[i] = float('inf')
            else:
                # Calculate curvature radius
                # R = s/θ where s is the arc length approximated by average of segment lengths
                s = (v1_norm + v2_norm) / 2
                radii[i] = s / theta
        
        # Handle endpoints
        radii[0] = radii[1]
        radii[-1] = radii[-2]
        
        return radii

def generate_reference_path(surface, step_length=25.0, num_points=10):
    """
    Generate reference path with enforced step length between points.
    """
    # Start at center of surface
    current_point = surface.evaluate(0, 0)
    reference_points = [current_point]
    
    # Define base direction (along v/y direction of surface)
    base_direction = np.array([0, 1, 0])
    
    # Generate points with controlled step length
    for i in range(num_points - 1):
        # Calculate next point ensuring step_length distance
        direction = base_direction + np.array([0.1 * np.sin(i * np.pi/5), 0, 0])
        direction = direction / np.linalg.norm(direction)
        
        # Calculate next raw point
        next_point_raw = current_point + direction * step_length
        
        # Project onto surface
        next_point = surface.evaluate(next_point_raw[0], next_point_raw[1])
        
        # Adjust to maintain step_length
        vec_to_next = next_point - current_point
        vec_to_next = vec_to_next / np.linalg.norm(vec_to_next) * step_length
        next_point = current_point + vec_to_next
        
        reference_points.append(next_point)
        current_point = next_point
    
    return np.array(reference_points)

def calculate_initial_shift_point(P0, P1, surface, tow_width):
    """
    Calculate Q0 ensuring exact tow_width distance from P0.
    """
    # Get path direction
    path_direction = P1 - P0
    path_direction = path_direction / np.linalg.norm(path_direction)
    
    # Get surface normal
    normal = surface.normal(P0[0], P0[1])
    normal = normal / np.linalg.norm(normal)
    
    # Calculate offset direction (perpendicular to both path and normal)
    offset_direction = np.cross(path_direction, normal)
    offset_direction = offset_direction / np.linalg.norm(offset_direction)
    
    # Calculate Q0 raw with exact tow_width distance
    Q0_raw = P0 + offset_direction * tow_width
    
    # Project onto surface
    Q0 = surface.evaluate(Q0_raw[0], Q0_raw[1])
    
    # Fine-tune to maintain exact tow_width distance
    vec_to_Q0 = Q0 - P0
    vec_to_Q0 = vec_to_Q0 / np.linalg.norm(vec_to_Q0) * tow_width
    Q0 = P0 + vec_to_Q0
    
    return Q0

def main():
    # Initialize parameters with correct relationships
    surface = Surface()
    step_length = 20.0  # mm
    surface.segment_length = step_length  # d = step_length
    surface.tow_width = 2 * surface.segment_length  # w = 2d
    
    # Generate reference path
    num_points=10
    reference_path = generate_reference_path(surface, step_length, num_points)

    analyze_path_curvature(reference_path)
    
    # Calculate initial Q0
    P0 = reference_path[0]
    P1 = reference_path[1]
    Q0 = calculate_initial_shift_point(P0, P1, surface, surface.tow_width)
    
    # Generate shifted path
    strip_model = PinJointedStrip(surface, surface.tow_width, surface.segment_length)
    Q_points = strip_model.propagate_path(reference_path, Q0)
    
    # Visualize
    if len(Q_points) >= 2:
        analyzer = CTSAnalyzer(surface)
        analyzer.visualize_intersection(
            reference_path, 
            Q_points, 
            None, None, None,
            reference_path,
            show_spheres=False,
            num_display_q=len(Q_points)
        )
        print("\nNot enough Q points for visualization")

if __name__ == "__main__":
    main()