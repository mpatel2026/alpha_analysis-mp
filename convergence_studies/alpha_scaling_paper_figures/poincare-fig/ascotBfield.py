import desc.io as dscio
import desc.grid as dscg
from desc.compat import rescale
from desc.grid import QuadratureGrid
from desc.magnetic_fields import PlasmaField
from desc.coils import MixedCoilSet

def desc_field_extended(fn: str, fn_encircling: str, fn_shaping: str, nphi: int=50, ntheta: int=120, 
                   nr: int=200, nz: int=200, 
                   psipad: float=0.000, waitingbar: bool=False,
                   rescale_R: float=None, rescale_B: float=None,
                   L_radial: int=4, M_poloidal: int=4, use_mixed_field: bool=False, 
                   use_stell_sym: bool=True, wall_offset: float=100) -> tuple[str, dict]:
        """Load magnetic field data extended beyond last closed flux surface
         from DESC equilibrium and coil files.

        This will behave like a template for the DESC equilibrium for the ASCOT5
        code: a dictionary with the inputs for the write_hdf5 function in the B_STS 
        ASCOT5 input.
        
        This routine allows for non-self-consistent rescaling of the equilibrium
        major radius R0 and magnetic field B0, which can be useful for simplified
        scans of the equilibrium parameters. 

        The radial and poloidal resolution of the DESC equilibrium used to compute
        the field on the concentric grid can be controlled with the L_radial and
        M_poloidal parameters, which act as multipliers of the original equilibrium
        resolution. Keep this numbers relatively higher (e.g. 4) to ensure good accuracy
        of the interpolated field, but take into account that the computational cost
        increases with them.

        The option to use_stell_sym allows to take advantage of stellarator symmetry
        in the equilibrium, which can significantly speed up the field computation.

        Parameters
        ----------
        fn : str
            File path to DESC equilibria HDF5 output.
        fn_encircling ; str
            File path to DESC encircling coil file
        fn_shaping ; str
            File path to DESC shaping coil file
        nphi : int, optional
            Number of toroidal angle phi grid points. Default = 360.
        ntheta : int, optional
            Number of poloidal angle theta grid points. Default = 120.
        nr : int, optional
            Number of radial coordinate R grid points. Default = 100.
        nz : int, optional
            Number of vertical coordinate Z grid points. Default = 100.
        psipad : float, optional
            Value to pad the toroidal flux psi0 on the magnetic axis (Wb). Default = 0.0.
        waitingbar : bool, optional
            Whether to show a progress bar during interpolation. Default = False.
        rescale_R : float, optional
            If provided, rescales the equilibrium major radius R0 to this value (m).
        rescale_B : float, optional
            If provided, rescales the equilibrium magnetic field B0 to this value (T).
        L_radial : int, optional
            Multiplier for the equilibrium radial resolution when computing on the
            concentric grid. Default = 4.
        M_poloidal : int, optional
            Multiplier for the equilibrium poloidal resolution when computing on the
            concentric grid. Default = 4.
        use_stell_sym : bool, optional
            Whether to use stellarator symmetry when computing the field. Default = True.
        wall_offset : float, optional
            Offset distance between wall and LCFS. Default assumes a wall at the LCFS. Default = 0. 


        Returns
        -------
        out : dict
            Dictionary with the following items:
            - `'axis_nphi'`, `'b_nphi'`, `'psi_nphi'`: nphi
            - `'b_nr'`, `'psi_nr'`: nr
            - `'b_nz'`, `'psi_nz'`: nz
            - `'axis_phimin'`, `'b_phimin'`, `'psi_phimin'`: phimin (deg)
            - `'axis_phimax'`, `'b_phimax'`, `'psi_phimax'`: (phimax-phimin)*(nphi-1)/nphi (deg)
            - `'b_rmin'`, `'psi_rmin'`: minimum radial coordinate R of output grids (m)
            - `'b_rmax'`, `'psi_rmax'`: maximum radial coordinate R of output grids (m)
            - `'b_zmin'`, `'psi_zmin'`: minimum vertical coordinate Z of output grids (m)
            - `'b_zmax'`, `'psi_zmax'`: maximum vertical coordinate Z of output grids (m)
            - `'axis_r'`: R(phi) on the magnetic axis (m)
            - `'axis_z'`: Z(phi) on the magnetic axis (m)
            - `'psi0'`: toroidal magnetic flux on the magnetic axis (Wb)
            - `'psi1'`: toroidal magnetic flux through the last closed flux surface (Wb)
            - `'psi'`: toroidal magnetic flux psi(R,phi,Z) (Wb)
            - `'br'`: radial magnetic field B_R(R,phi,Z) (T)
            - `'bphi'`: toroidal magnetic field B_phi(R,phi,Z) (T)
            - `'bz'`: vertical magnetic field B_Z(R,phi,Z) (T)
            - `'Nperiods'`: Number of field periods in the equilibrium
            - `'stell_sym'`: whether to use stellarator symmetry (bool)
        """
        if not os.path.isfile(fn):
            raise FileNotFoundError(f"DESC file {fn} not found.")
        if not os.path.isfile(fn_encircling):
            raise FileNotFoundError(f"DESC file {fn_encircling} not found. Please provide encircling coil file")
        if not os.path.isfile(fn_shaping):
            raise FileNotFoundError(f"DESC file {fn_shaping} not found. Please provide shaping coil file")

        if wall_offset < 0:
            raise ValueError("Wall offset must be >= 0.")
        
        if not hasattr(wall_offset, 'units'):
            # Assume it was meant to be cm if no units provided
            wall_offset = wall_offset * unyt.cm
        else:
            wall_offset = wall_offset.to(unyt.cm)

        fam = dscio.load(fn, file_format="hdf5")
        try:  # if file is an EquilibriaFamily, use final Equilibrium
            eq = fam[-1]
        except:  # file is already an Equilibrium
            eq = fam

        if (rescale_R is not None) and (rescale_B is not None):
            eq = rescale(eq, L=("R0", rescale_R), B=("B0", rescale_B))
        elif rescale_R is not None:
            eq = rescale(eq, L=("R0", rescale_R))
        elif rescale_B is not None:
            eq = rescale(eq, B=("B0", rescale_B))

        # toroidal angle array
        # Patch for the abscense of the units decorators.
        phimin = 0.0 * unyt.deg
        phimax = 360.0 * unyt.deg / eq.NFP # Number of field periods.

        if use_stell_sym:
            phimax /= 2.0 # We can further reduce the toroidal angle range.

            # This padding is added to remove the edge effects since the
            # implemented symmetry does not allow us to use the symmetric
            # spline in ASCOT.
            dphi = (phimax - phimin) / (nphi - 1)
            phimin = - 4.0 * dphi
            phimax = phimax + 4.0 * dphi
            nphi += 8

        phi = np.linspace(phimin.to('rad').value, 
                          phimax.to('rad').value, 
                          nphi, endpoint=True)  # rad
        # note: phi should start at 0 and end on 360, inclusive

        # magnetic axis
        grid_axis = dscg.LinearGrid(rho=0.0, zeta=nphi, NFP=1)
        data_axis = eq.compute(["R", "Z"], grid=grid_axis)
        axis_r = data_axis["R"]  # m
        axis_z = data_axis["Z"]  # m
        psi0 = 0  # Wb

        # boundary
        grid = dscg.LinearGrid(
            rho=1.0, theta=ntheta, zeta=nphi, NFP=1, sym=False, endpoint=True
        )
        data = eq.compute(["R", "Z"], grid=grid)
        bdry_r = data["R"].reshape((grid.num_zeta, grid.num_theta), order="C").T * unyt.m
        bdry_z = data["Z"].reshape((grid.num_zeta, grid.num_theta), order="C").T * unyt.m

        # boundaries
        # NOTE For adding wall offsets, in order to get rz grid extended outwards, decrease/increase the bounds by wall_offset times a small multiplier (e.g. 1.2) to give a cushion for bfield beyond wall.
        bfield_offset = wall_offset * 1.2
        print("bfield_offset", bfield_offset)
        rmin = np.min(bdry_r) - bfield_offset  # m
        rmax = np.max(bdry_r) + bfield_offset  # m
        zmin = np.min(bdry_z) - bfield_offset  # m
        zmax = np.max(bdry_z) + bfield_offset  # m
        psi1 = eq.Psi * unyt.Wb  # Wb

        R_1d = np.linspace(rmin, rmax, nr)  # m
        Z_1d = np.linspace(zmin, zmax, nz)  # m
        Z_2d, R_2d = np.meshgrid(Z_1d, R_1d)
        if hasattr(Z_2d, 'units'):
            Z_2d = Z_2d.to('m').value
        if hasattr(R_2d, 'units'):
            R_2d = R_2d.to('m').value

        # interpolate psi, B_R, B_phi, B_Z to cylindircal coordinates
        psi = np.zeros([nr, nz, nphi]) * unyt.Wb
        psi_extended = np.zeros([nr, nz, nphi]) * unyt.Wb
        br = np.zeros([nr, nz, nphi]) * unyt.T
        bphi = np.zeros([nr, nz, nphi]) * unyt.T
        bz = np.zeros([nr, nz, nphi]) * unyt.T

        #load in coils for coil current bfield calculation
        encircling = dscio.load(fn_encircling)
        shaping = dscio.load(fn_shaping)
        coils = MixedCoilSet((encircling, shaping), check_intersection=False)
        # source grid is used to compute the vector potential from the plasma current density
        source_grid = QuadratureGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
        # create a PlasmaField object to perform the vector potential calculation
        time_start_PlasmaField = time.time()
        print('computing plasma field')
        #NOTE Changing increasing A_res below can lead to high fidelity bfields across the lcfs
        field = PlasmaField(
            eq,
            source_grid=source_grid,
            R_bounds=(rmin, rmax),  # R bounds of the computational domain
            Z_bounds=(zmin, zmax),  # Z bounds of the computational domain
            A_res=128,  # resolution of vector potential A, higher resolution is better (128 is good)
        )
        time_end_PlasmaField = time.time()
        print(f'finished plasma field in {time_end_PlasmaField - time_start_PlasmaField:.2f} seconds')

        # 1. Define a helper function and JIT it.
        # This tells JAX: "Learn the SHAPE of this math, not the specific values."
        @jax.jit
        def get_coil_bfield(coords_batch):
            # Pass the static chunk_size here
            return coils.compute_magnetic_field(coords_batch, source_grid=None, chunk_size=10000)
        

        # 2. Pre-prepare your static R and Z data once (outside the loop)
        r_jax = jnp.array(R_2d.ravel())
        z_jax = jnp.array(Z_2d.ravel())

        for k in tqdm(range(nphi), desc="Interpolating DESC field", total=nphi, disable=not waitingbar):
            psi1 = eq.Psi * unyt.Wb  # Wb
            # compute on concentric grid
            grid = dscg.ConcentricGrid(
                L=eq.L_grid*L_radial, M=eq.M_grid*M_poloidal, N=0, 
                NFP=eq.NFP, node_pattern="linear"
            )
            if hasattr(phi, 'units'):
                iphi = phi[k].to('rad').value
            else:
                iphi = phi[k]
            grid._nodes[:, 2] = iphi
            data = eq.compute(["R", "Z", "psi", "B_R", "B_phi", "B_Z"], grid=grid)
            R = data["R"]
            Z = data["Z"]            
            psi_data = data["psi"]
            psi_data_2pi = psi_data * 2 * np.pi
            psi[:, :, k] = griddata(
            (R, Z),
            psi_data_2pi,
            (R_2d, Z_2d),
            fill_value=psi1)

            #Compute coil current contributions to b_field.            
            phi_jax = jnp.full_like(r_jax, iphi)
            coords_jax = jnp.column_stack([r_jax, phi_jax, z_jax])
            bfield_coils = get_coil_bfield(coords_jax)

            br_coil = bfield_coils[:,0].reshape(nr,nz) 
            bphi_coil = bfield_coils[:,1].reshape(nr,nz)
            bz_coil = bfield_coils[:,2].reshape(nr,nz)

                    
            bfield_plasma = field.compute_magnetic_grid(R_1d, iphi, Z_1d, eq.NFP).reshape(-1, 3)
            B_R_plasma = bfield_plasma[:, 0].reshape(nr,nz)
            B_phi_plasma = bfield_plasma[:, 1].reshape(nr,nz)
            B_Z_plasma = bfield_plasma[:, 2].reshape(nr,nz) 

            target_pts = np.vstack([R_2d.ravel(), Z_2d.ravel()]).T

            interp_br_plasma = RegularGridInterpolator((R_1d, Z_1d), np.asarray(B_R_plasma), bounds_error=False, fill_value=0)
            interpolated_values_br_plasma = interp_br_plasma(target_pts)
            br_plasma = interpolated_values_br_plasma.reshape(R_2d.shape)

            interp_bphi_plasma = RegularGridInterpolator((R_1d, Z_1d), np.asarray(B_phi_plasma), bounds_error=False, fill_value=0)
            interpolated_values_bphi_plasma = interp_bphi_plasma(target_pts)
            bphi_plasma = interpolated_values_bphi_plasma.reshape(R_2d.shape)            

            interp_bz_plasma = RegularGridInterpolator((R_1d, Z_1d), np.asarray(B_Z_plasma), bounds_error=False, fill_value=0)
            interpolated_values_bz_plasma = interp_bz_plasma(target_pts)
            bz_plasma = interpolated_values_bz_plasma.reshape(R_2d.shape)

            br_total = (br_coil + br_plasma) * unyt.T
            bphi_total = (bphi_coil + bphi_plasma) * unyt.T 
            bz_total = (bz_coil + bz_plasma) * unyt.T
            #get points inside lcfs by finding points where psi < ps1, and then set those points equal to eq.compute values of B
            if use_mixed_field:
                br_lcfs = griddata((R, Z), data["B_R"], (R_2d, Z_2d)) * unyt.T
                bphi_lcfs = griddata((R, Z), data["B_phi"], (R_2d, Z_2d)) * unyt.T
                bz_lcfs = griddata((R, Z), data["B_Z"], (R_2d, Z_2d)) * unyt.T
                inside_lcfs = psi[:, :, k] < (psi1.value - 1) * unyt.Wb # add a small buffer to ensure we are safely inside lcfs for these points
                br_total[inside_lcfs] = br_lcfs[inside_lcfs]
                bphi_total[inside_lcfs] = bphi_lcfs[inside_lcfs]
                bz_total[inside_lcfs] = bz_lcfs[inside_lcfs]

            #Add coil and plasma current contributions for total bfield
            br[:, :, k] = br_total
            bphi[:, :, k] = bphi_total
            bz[:, :, k] = bz_total

            # Replace nan's with closest value
            data = br[:, :, k].to('T').value
            mask = np.where(~np.isnan(data))
            interp = NearestNDInterpolator(np.transpose(mask), data[mask])
            filled_data = interp(*np.indices(data.shape))
            br[:, :, k] = filled_data * unyt.T

            data = bz[:, :, k].to('T').value
            mask = np.where(~np.isnan(data))
            interp = NearestNDInterpolator(np.transpose(mask), data[mask])
            filled_data = interp(*np.indices(data.shape))
            bz[:, :, k] = filled_data * unyt.T


            data = bphi[:, :, k].to('T').value
            mask = np.where(~np.isnan(data))
            interp = NearestNDInterpolator(np.transpose(mask), data[mask])
            filled_data = interp(*np.indices(data.shape))
            bphi[:, :, k] = filled_data * unyt.T
            #clear cache to prevent memory build up of coils.compute_magnetic_grid
            if k % 50 == 0 and k > 0:
                jax.clear_caches()

        # change order from [R,Z,phiang] to [R,phiang,Z]
        br = np.transpose(br, (0, 2, 1))
        bphi = np.transpose(bphi, (0, 2, 1))
        bz = np.transpose(bz, (0, 2, 1))
        out = {
            "axis_phimin": 0.0, 
            "axis_phimax": 2*np.pi, 
            "axis_nphi": nphi,
            "axisr": axis_r,  # m
            "axisz": axis_z,  # m
            "b_rmin": rmin,  # m
            "b_rmax": rmax,  # m
            "b_nr": nr,
            "b_zmin": zmin,  # m
            "b_zmax": zmax,  # m
            "b_nz": nz,
            "b_phimin": phimin,  # deg
            "b_phimax": np.rad2deg(phi[-1]),  # deg
            "b_nphi": nphi,
            "br": br,  # T
            "bphi": bphi,  # T
            "bz": bz,  # T
            "psi": psi_extended,  # Wb
            "psi0": psi0,  # Wb
            "psi1": eq.Psi * unyt.Wb,  # Wb
            "psi_rmin": rmin,  # m
            "psi_rmax": rmax,  # m
            "psi_nr": nr,
            "psi_zmin": zmin,  # m
            "psi_zmax": zmax,  # m
            "psi_nz": nz,
            "psi_phimin": phimin,  # deg
            "psi_phimax": np.rad2deg(phi[-1]),  # deg
            "psi_nphi": nphi,
            "Nperiods": eq.NFP,
            "stell_sym": use_stell_sym,  # m
        }

        return ('B_STS', out)