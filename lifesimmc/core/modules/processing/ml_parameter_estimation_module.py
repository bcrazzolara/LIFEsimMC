import numpy as np
import torch
from lmfit import minimize, Parameters
from rich.console import Console

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.planet_resource import PlanetResource
from lifesimmc.core.resources.resource_collection import ResourceCollection


class MLSEDEstimationModule(BaseModule):
    """Class representation of a module that performs maximum likelihood estimation (MLE) of the planet's SED.

    Parameters
    ----------
    n_setup_in : str
        Name of the input configuration resource.
    n_data_in : str
        Name of the input data resource.
    n_planets_out : str
        Name of the output planet parameters resource.
    n_transformation_in : str, optional
        Name of the input transformation resource. If None, no transformation is applied.
    n_template_in : str, optional
        Name of the input template resource. If None, no template is used.

    Attributes
    ----------
    cost_function : torch.Tensor or None
        The full field-of-view grid search statistic from the analytical initial guess (shape
        grid_size x grid_size), available after run() -- e.g. `module.cost_function` -- for
        inspecting the position likelihood surface directly (peak location, sidelobes, etc.)
        rather than just the single best-fit point. None if n_planets_in was supplied (skips
        the grid search entirely and uses the given planet's position/SED as the initial guess).

    With fix_position=True (requires n_planets_in), the planet position is held fixed at the position of the input
    planet and only the fluxes are fitted. The returned SED, std and cov (n_wavelengths x n_wavelengths) are then
    conditional on the known position, i.e. the 'known position' case.
    """

    def __init__(
            self,
            n_setup_in: str,
            n_data_in: str,
            n_planets_out: str,
            n_transformation_in: str = None,
            n_template_in: str = None,
            n_planets_in: str = None,
            bounds: bool = False,
            fix_position: bool = False
    ):
        """Constructor method.

        Parameters
        ----------
        n_setup_in : str
            Name of the input configuration resource.
        n_data_in : str
            Name of the input data resource.
        n_planets_out : str
            Name of the output planet parameters resource.
        n_transformation_in : str, optional
            Name of the input transformation resource. If None, no transformation is applied.
        n_template_in : str, optional
            Name of the input template resource. If None, no template is used. # TODO: handle no templates
        n_planets_in : str, optional
            Name of the input planet resource. If given, its SED and position are used as the initial guess instead of
            the grid search.
        bounds : bool, optional
            If True, the fluxes are constrained to be non-negative.
        fix_position : bool, optional
            If True, the position is fixed to that of n_planets_in and only the fluxes are fitted (known position).
        """
        if fix_position and n_planets_in is None:
            raise ValueError("fix_position=True requires n_planets_in to provide the known planet position.")
        super().__init__()
        self.n_setup_in = n_setup_in
        self.n_data_in = n_data_in
        self.n_template_in = n_template_in
        self.n_transformation_in = n_transformation_in
        self.n_planets_out = n_planets_out
        self.n_planets_in = n_planets_in
        self.bounds = bounds
        self.fix_position = fix_position

    def _get_analytical_initial_guess(self, data, template_data, grid_coordinates):
        # Normalize template data with their variance along axis 2
        # template_data = template_data / torch.var(template_data, axis=2, keepdim=True) ** 0.5

        # Calculate matrix C according to equation B.2
        data_variance = torch.var(data, axis=0)
        sum = torch.sum(torch.einsum('ij, ijkl->ijkl', data, template_data), axis=0)
        vector_c = torch.einsum('ijk, i->ijk', sum, 1 / data_variance)

        # Calculate matrix B according to equation B.3
        sum = torch.sum(template_data ** 2, axis=0)
        vector_b = torch.nan_to_num(torch.einsum('ijk, i->ijk', sum, 1 / data_variance), 1)

        # Create diagonal matrix B
        b_shape = vector_b.shape
        eye = torch.eye(b_shape[0], device=self.device).unsqueeze(-1).unsqueeze(-1)
        x_diag = vector_b.unsqueeze(1)
        matrix_b = eye * x_diag

        # Calculate the optimum flux according to equation B.6 and set positivity constraint
        b_perm = matrix_b.permute(2, 3, 0, 1)
        b_inv = torch.linalg.inv(b_perm).permute(2, 3, 0, 1)

        # Optimum flux for each template
        optimum_flux = torch.einsum('ijkl, jkl->ikl', b_inv, vector_c)

        # Calculate the cost function according to equation B.8
        optimum_flux = torch.where(optimum_flux >= 0, optimum_flux, 0)
        cost_function = optimum_flux * vector_c
        cost_function = torch.sum(torch.nan_to_num(cost_function, 0), axis=0)

        # Get the optimum flux at the position of the maximum of the cost function
        flat_idx = cost_function.argmax()  # scalar
        row = flat_idx // cost_function.shape[1]
        col = flat_idx % cost_function.shape[1]
        optimum_flux_at_maximum = optimum_flux[:, row.item(), col.item()].cpu().numpy()  # TODO: fix this scaling

        # Get the coordinates of the maximum
        x_coord = grid_coordinates[0][row.item(), col.item()].cpu().numpy()
        y_coord = grid_coordinates[1][row.item(), col.item()].cpu().numpy()

        return optimum_flux_at_maximum, x_coord, y_coord, cost_function

    def run(self, pipeline_resources: list[BaseResource | ResourceCollection]) -> tuple[
        ResourceCollection[PlanetResource]]:
        console = Console()

        with console.status("Estimating SED using numerical maximum likelihood estimation...", spinner="dots"):
            r_setup_in = self.get_resource_from_name(self.n_setup_in)
            r_templates_in = self.get_resource_from_name(self.n_template_in)
            r_transformation_in = self.get_resource_from_name(self.n_transformation_in) \
                if self.n_transformation_in \
                else None
            planets_in = self.get_resource_from_name(self.n_planets_in) if self.n_planets_in else None

            transf_in = r_transformation_in.transformation if r_transformation_in else lambda x: x
            data_in = self.get_resource_from_name(self.n_data_in).get_data()
            template_data_in = r_templates_in.get_data()
            grid_coordinates = r_templates_in.grid_coordinates

            # Flatten data along differential outputs and times axes
            nk, nl, nt = data_in.shape
            _, _, _, ng, _ = template_data_in.shape

            data_in = data_in.permute(0, 2, 1).reshape(nk * nt, nl)

            template_data_in = (
                template_data_in
                .permute(0, 2, 1, 3, 4)
                .reshape(nk * nt, nl, ng, ng)
            )

            # Set up parameters and initial conditions
            if planets_in is None:
                sed_init, posx_init, posy_init, self.cost_function = self._get_analytical_initial_guess(
                    data_in,
                    template_data_in,
                    grid_coordinates
                )
            # If planet_params_in is provided, use its values as initial conditions
            else:
                # TODO: implement for multiple planets
                sed_init = planets_in.collection[0].planet.spectral_energy_distribution[:, 0, 0].cpu().numpy()
                posx_init = planets_in.collection[0].planet.sky_coordinates[0, 0, 0, 0, 0].cpu().numpy()
                posy_init = planets_in.collection[0].planet.sky_coordinates[1, 0, 0, 0, 0].cpu().numpy()
                # No grid search was run, so there's no cost-function map for this call.
                self.cost_function = None

            data_in = data_in.cpu().numpy()
            hfov_max = r_setup_in.phringe.get_field_of_view()[-1].cpu().numpy() / 2

            # Prepare initial parameter set guess for the minimizer
            params = Parameters()

            for j in range(len(sed_init)):
                if self.bounds:
                    params.add(f'flux_{j}', value=sed_init[j], min=0)
                else:
                    params.add(f'flux_{j}', value=sed_init[j])
            params.add('pos_x', value=posx_init, min=-hfov_max, max=hfov_max, vary=not self.fix_position)
            params.add('pos_y', value=posy_init, min=-hfov_max, max=hfov_max, vary=not self.fix_position)

            # Perform MLE
            def residual_data(params, target):
                posx = params['pos_x'].value
                posy = params['pos_y'].value
                flux = np.array([params[f'flux_{z}'].value for z in range(len(sed_init))])
                model = r_setup_in.phringe.get_model_counts(
                    spectral_energy_distribution=flux,
                    x_position=posx,
                    y_position=posy,
                    kernels=True
                )
                model = transf_in(model)
                model = np.transpose(model, (0, 2, 1))
                model = model.reshape(data_in.shape)

                return model - target

            out = minimize(residual_data, params, args=(data_in,), method='leastsq')
            cov_out = out.covar

            # lmfit scales covar by redchi = chisqr / (ndata - nvarys), where ndata is the raw residual length.
            # Transformations such as ZCA whitening's mean subtraction (zca_whitening_module.py) can remove degrees
            # of freedom from the residual; rescale to the corrected redchi so stderr isn't underestimated.
            dof_lost = r_transformation_in.dof_lost if r_transformation_in else 0
            if cov_out is not None and out.nfree > dof_lost:
                cov_out = cov_out * out.nfree / (out.nfree - dof_lost)

            fluxes = np.array([out.params[f'flux_{k}'].value for k in range(len(sed_init))])
            posx = out.params['pos_x'].value
            posy = out.params['pos_y'].value

            # lmfit's covar only covers the varied parameters, so with a fixed position it is fluxes-only
            try:
                stds = np.sqrt(np.diag(cov_out))
                if self.fix_position:
                    flux_err = stds
                    posx_err = 0.0
                    posy_err = 0.0
                else:
                    flux_err = stds[0:-2]
                    posx_err = stds[-2]
                    posy_err = stds[-1]
            except ValueError:
                flux_err = np.full_like(fluxes, np.nan)
                posx_err = np.nan
                posy_err = np.nan

            # TODO: Implement multi-planet signal extraction
            r_planets_out = ResourceCollection[PlanetResource](
                name=self.n_planets_out,
            )
            planet_resource = PlanetResource(
                name='',
                planet=None,
                sed=fluxes,
                std=flux_err,
                cov=cov_out,
                pos_x=posx,
                pos_y=posy
            )
            r_planets_out.collection.append(planet_resource)

        print('Done')
        return r_planets_out,
