# -------------------------
# Classe utilitaire: LossVisualization
# -------------------------
from matplotlib import pyplot as plt
import numpy as np
import torch


class LossVisualization:
    """Classe pour construire une coupe 2D de la surface de loss autour d'un point
    de paramètres et projeter la trajectoire d'entraînement.

    Utilisation typique:
        lv = LossVisualization(
            model_class=SimpleMLP,
            set_param_fn=set_param_vector,
            criterion=criterion,
            X_val=X_val_t,
            y_val=y_val_t,
            trajectory=trajectory,
            base_vec=base_vec,
            radius=RADIUS,
            grid_n=GRID_N,
            surface_cmap=SURFACE_CMAP,
            contour_cmap=CONTOUR_CMAP,
            traj_color=TRAJECTORY_COLOR,
            traj_marker=TRAJECTORY_MARKER,
        )
        lv.compute_directions()
        lv.evaluate_grid()
        lv.project_trajectory()
        lv.plot()
    """
    def __init__(self,
                 model_class,
                 set_param_fn,
                 criterion,
                 X_val,
                 y_val,
                 trajectory,
                 base_vec=None,
                 radius=0.6,
                 grid_n=61,
                 surface_cmap='plasma',
                 contour_cmap='plasma',
                 traj_color='black',
                 traj_marker='o',
                 model_kwargs=None,
                 device='cpu'):
        self.model_class = model_class
        self.set_param_fn = set_param_fn
        self.criterion = criterion
        self.X_val = X_val
        self.y_val = y_val
        self.trajectory = trajectory
        self.base_vec = base_vec if base_vec is not None else trajectory[-1].clone()
        self.radius = radius
        self.grid_n = grid_n
        self.surface_cmap = surface_cmap
        self.contour_cmap = contour_cmap
        self.traj_color = traj_color
        self.traj_marker = traj_marker
        self.model_kwargs = model_kwargs or {}
        self.device = device

        # placeholders
        self.d1 = None
        self.d2 = None
        self.alphas = None
        self.betas = None
        self.loss_grid = None
        self.proj_a = None
        self.proj_b = None
        self.traj_heights = None

    def compute_directions(self, seed=None):
        """Génère deux directions orthonormées d1,d2 dans l'espace des paramètres."""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        param_dim = self.base_vec.numel()
        d1 = torch.randn(param_dim)
        d2 = torch.randn(param_dim)

        d1 = d1 / d1.norm()
        d2 = d2 - (d2 @ d1) * d1
        if d2.norm().item() == 0:
            d2 = torch.randn(param_dim)
            d2 = d2 - (d2 @ d1) * d1
        d2 = d2 / d2.norm()

        d1 *= self.radius
        d2 *= self.radius

        self.d1 = d1
        self.d2 = d2
        return d1, d2

    def evaluate_grid(self, overwrite=True):
        """Calcule la loss sur la grille (alphas x betas) autour de base_vec."""
        assert self.d1 is not None and self.d2 is not None, "call compute_directions() first"
        self.alphas = np.linspace(-1.0, 1.0, self.grid_n)
        self.betas = np.linspace(-1.0, 1.0, self.grid_n)
        loss_grid = np.zeros((self.grid_n, self.grid_n), dtype=float)

        model_tmp = self.model_class(**self.model_kwargs)
        model_tmp.to(self.device)

        with torch.no_grad():
            for i, a in enumerate(self.alphas):
                for j, b in enumerate(self.betas):
                    vec = self.base_vec + float(a) * self.d1 + float(b) * self.d2
                    self.set_param_fn(model_tmp, vec)
                    logits = model_tmp(self.X_val)
                    loss_grid[j, i] = float(self.criterion(logits, self.y_val).item())

        self.loss_grid = loss_grid
        return loss_grid

    def project_trajectory(self):
        """Projette la trajectoire (liste de vecteurs) sur d1,d2 unitaires et calcule
        les hauteurs (loss) le long de la trajectoire.
        """
        assert self.d1 is not None and self.d2 is not None, "call compute_directions() first"
        d1_unit = self.d1 / self.d1.norm()
        d2_unit = self.d2 / self.d2.norm()

        proj_a = []
        proj_b = []
        for tvec in self.trajectory:
            delta = tvec - self.base_vec
            proj_a.append(float(delta @ d1_unit))
            proj_b.append(float(delta @ d2_unit))
        proj_a = np.array(proj_a)
        proj_b = np.array(proj_b)

        traj_heights = []
        model_tmp = self.model_class(**self.model_kwargs)
        model_tmp.to(self.device)
        with torch.no_grad():
            for a, b in zip(proj_a, proj_b):
                vec = self.base_vec + a * d1_unit + b * d2_unit
                self.set_param_fn(model_tmp, vec)
                traj_heights.append(float(self.criterion(model_tmp(self.X_val), self.y_val).item()))

        self.proj_a = proj_a
        self.proj_b = proj_b
        self.traj_heights = np.array(traj_heights)
        return proj_a, proj_b, self.traj_heights

    def plot(self, figsize=(13,6), levels=200, start_marker_color='blue'):
        """Trace la surface 3D, la trajectoire et le contour 2D."""
        assert self.loss_grid is not None, "call evaluate_grid() first"
        assert self.proj_a is not None and self.proj_b is not None and self.traj_heights is not None, "call project_trajectory() first"

        A, B = np.meshgrid(self.alphas, self.betas)
        fig = plt.figure(figsize=figsize)

        ax = fig.add_subplot(121, projection='3d')
        ax.plot_surface(A, B, self.loss_grid, cmap=self.surface_cmap, linewidth=0, antialiased=True, alpha=0.95)
        ax.set_xlabel('alpha (d1)')
        ax.set_ylabel('beta (d2)')
        ax.set_zlabel('loss')
        ax.set_title('Loss surface (autour de w*)')

        # start point (premier point de la trajectoire)
        ax.plot([self.proj_a[0]], [self.proj_b[0]], [self.traj_heights[0]], color=start_marker_color, marker=self.traj_marker, markersize=6, label='Start')
        # trajectoire complète
        ax.plot(self.proj_a, self.proj_b, self.traj_heights, color=self.traj_color, marker=self.traj_marker, markersize=3, linewidth=2, label='trajectory')
        ax.legend()

        ax2 = fig.add_subplot(122)
        CS = ax2.contourf(A, B, self.loss_grid, levels=levels, cmap=self.contour_cmap)
        ax2.plot([self.proj_a[0]], [self.proj_b[0]], '-'+self.traj_marker, color='r', markersize=8)
        ax2.plot(self.proj_a, self.proj_b, '-'+self.traj_marker, color=self.traj_color, markersize=3)
        ax2.set_xlabel('alpha (d1)')
        ax2.set_ylabel('beta (d2)')
        ax2.set_title('Contour du loss (vue de dessus)')
        fig.colorbar(CS, ax=ax2, label='loss')

        plt.tight_layout()
        plt.show()