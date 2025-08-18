import numpy as np
from matplotlib.axes import Axes
from matplotlib import colormaps
from matplotlib import pyplot as plt
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.ext.matproj import MPRester
from .electron_atomic_form_factor import default_electron_atomic_form_factors


class EwaldSphere:
    hbar_c = 197 # eV-nm
    m0 = 511e3 # eV
    def __init__(self, energy_eV:float):
        self.energy_eV = energy_eV
    @property
    def wavelength(self):
        '''Get the wavelength in nm'''
        return 2*np.pi*self.hbar_c / np.sqrt(self.energy_eV**2 + 2*self.m0*self.energy_eV)
    @property
    def k_z(self):
        '''Get the wavevector in cyc/nm, crystallographic convention'''
        return 1/self.wavelength
    @property
    def k_z_angular(self):
        '''Get the wavevector in rad/nm, physics convention'''
        return 2*np.pi*self.k_z
    def sg(self, g:tuple[float,float,float], kx:np.ndarray=0, ky:np.ndarray=0):
        '''Returns the excitation error of a given point in reciprocal space'''
        sg = g[2] + (kx**2 + ky**2 - (kx+g[0])**2 - (ky+g[1])**2) / (2*self.k_z)
        return sg
    def sg_en_masse(self, g:np.ndarray, kx:np.ndarray=0, ky:np.ndarray=0):
        '''Returns the excitation error of a provided points in reciprocal space.
        
        Args:
            g (np.ndarray): An array of reciprocal lattice vectors, of shape (N,3), in nm^-1
            kx, ky (float): Small beam tilts, in nm^-1
        
        Returns:
            sg (np.ndarray): An array of excitation errors corresponding to the provided reciprocal lattice vectors. Shape (N,)
        '''
        sg = g[:,2] + (kx**2 + ky**2 - (kx+g[:,0])**2 - (ky+g[:,1])**2) / (2*self.k_z)
        return sg


class StructureHandler:
    MATERIALS_PROJECT_API = 'E3sCeYTbOOxpD9gtJ1EgevVyhHUn3w2J'
    COMMON_MATERIAL_IDS = dict(Si_Fd3m      = 'mp-149',     # diamond cubic Si
                               Se_Pm3m      = 'mp-7755',    # simple cubic Se, the only known single-element simple cubic structure
                               HfO2_Pca21   = 'mp-685097',  # ferroelectric orthorhombic HfO2
                               HfO2_P21c    = 'mp-352',     # monoclinic HfO2
                               HfO2_P42nmc  = 'mp-1018721') # tetragonal HfO2
    @classmethod
    def from_matproj(cls, material_id:str="Si_Fd3m", conventional_unit_cell=True):
        if material_id in cls.COMMON_MATERIAL_IDS.keys():
            material_id = cls.COMMON_MATERIAL_IDS[material_id]
        with MPRester(cls.MATERIALS_PROJECT_API) as mpr:
            return cls(mpr.get_structure_by_material_id(material_id=material_id, 
                                                        conventional_unit_cell=conventional_unit_cell))
    def __init__(self, struct:Structure):
        '''Here surface normal are the [uvw] (fractional coordinates) to use for aligning the Structure's surface normal to the incident beam.
           This is 
        '''
        self.struct = struct
    def align_structure(self, 
                        beam_incident:tuple[float,float,float]=(0,0,1),
                        beam_x_ref:tuple[float,float,float]=(1,0,0)):
        """
        Rotates a Structure so that:
        - beam_incident (uvw) is along -z in the lab frame
        - beam_x_ref (uvw) is along +x in the lab frame. This will be orthogonalized relative to the incident beam.
        
        This is done by rotating the lattice vectors and keeping the fractional coordinates of the basis atoms.

        Args:
            beam_incident (tuple/list): UVW direction for the beam (real-space lattice coords)
            beam_x_ref (tuple/list): UVW direction defining +x in lab frame (real-space lattice coords)
        """

        # Step 1: Convert UVW to Cartesian directions
        lat_matrix = np.array(self.struct.lattice.matrix)  # 3x3 matrix
        beam_cart = np.dot(beam_incident, lat_matrix)
        x_ref_cart = np.dot(beam_x_ref, lat_matrix)

        # Step 2: Normalize beam direction (beam along -z in lab frame)
        z_lab = -beam_cart / np.linalg.norm(beam_cart)

        # Step 3: Remove any component of x_ref along z_lab, normalize
        x_lab = x_ref_cart - np.dot(x_ref_cart, z_lab) * z_lab
        x_lab /= np.linalg.norm(x_lab)

        # Step 4: y_lab = z × x
        y_lab = np.cross(z_lab, x_lab)

        # Step 5: Build rotation matrix (crystal→lab)
        R = np.vstack([x_lab, y_lab, z_lab]).T  # Columns are lab axes in crystal Cartesian coords

        # Step 6: Rotate lattice vectors
        new_lattice_matrix = np.dot(lat_matrix, R)

        # Step 7: Return new Structure with rotated lattice
        self.struct = Structure(
            lattice=new_lattice_matrix,
            species=self.struct.species,
            coords=self.struct.frac_coords,
            coords_are_cartesian=False
        )
    def plot_unit_cell(self, low_frac:tuple[float,float,float]=(-0.1,-0.1,-0.1),
                             high_frac:tuple[float,float,float]=(1.1,1.1,1.1),
                             frame:bool=True, colors:dict=None, sizes:dict={'default':100},
                             origin:tuple=(0,0,0)):
        '''Implemented before I knew about https://pymatgen.org/pymatgen.vis.html#pymatgen.vis.structure_chemview.quick_view'''
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
        ax.set_title(f'${self.struct.get_space_group_info()[0]}$')
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        a_lo = low_frac[0]
        b_lo = low_frac[1]
        c_lo = low_frac[2]
        a_hi = high_frac[0]
        b_hi = high_frac[1]
        c_hi = high_frac[2]

        supercell = self.struct.make_supercell(3,in_place=False)

        origin = np.array(origin)
        # Plot the atoms
        for el in set(self.struct.species):
            all_x, all_y, all_z = [],[],[]
            for site in supercell:
                xf,yf,zf = self.struct.lattice.get_fractional_coords(site.coords) # go through sites in supercell, map them back to the single cell, keep only the atoms whose coords fall within the desired unit cell

                if (  a_lo<=xf<=a_hi and 
                      b_lo<=yf<=b_hi and
                      c_lo<=zf<=c_hi and 
                      el == site.specie  ):
                    x,y,z = (site.coords - origin)/10 # angstrom to nm
                    all_x.append(x)
                    all_y.append(y)
                    all_z.append(z)
            ax.scatter(all_x,
                       all_y,
                       all_z, 
                       label=el, 
                       s=sizes[el] if el in sizes else sizes['default'],
                       color=colors[el] if colors is not None else None)
        
        if frame:
            o = [0,0,0]
            a1,a2,a3 = self.struct.lattice.matrix/10 # angstrom to nm
            edges = [
                (o, a1),
                (o, a2),
                (o, a3),
                (a1, a1 + a2),
                (a1, a1 + a3),
                (a2, a2 + a1),
                (a2, a2 + a3),
                (a3, a3 + a1),
                (a3, a3 + a2),
                (a1 + a2, a1 + a2 + a3),
                (a1 + a3, a1 + a2 + a3),
                (a2 + a3, a1 + a2 + a3)
            ]
            for start, end in edges:
                xs, ys, zs = zip(start, end)
                ax.plot(xs, ys, zs, color='black')

        return fig, ax
    def get_hkl_family(self, hkl:tuple[int,int,int])->set[tuple[int,int,int]]:
        '''Get all members of the family of [hkl] equivalent by the symmetry of the provided pymatgen structure.'''

        hkl = tuple((int(x) for x in hkl))
        symops = SpacegroupAnalyzer(self.struct).get_symmetry_operations(cartesian=False)

        # Apply all symops and collect, ignoring duplicates
        equiv = set()
        for op in symops:
            hkl2 = op.operate(hkl)
            hkl2 = np.rint(hkl2).astype(int)
            equiv.add(tuple((int(hkl) for hkl in hkl2)))

        return equiv
    def are_hkl_in_same_family(self,
                        hkl1:tuple[int,int,int],
                        hkl2:tuple[int,int,int])->bool:
        '''Determine if two [hkl] in a given pymatgen Structure are related by symmetry, i.e. if they belong to the same family.'''

        assert len(hkl1) == len(hkl2)
        hkl2 = tuple((int(x) for x in hkl2))
        if hkl2 in self.get_hkl_family(hkl1):
            return True
        else:
            return False
    def get_excitable_hkl(self, ewald:EwaldSphere, max_sg:float, max_g:float=None, kx:float=0, ky:float=0, exclude_000:bool=True):
        """
        Returns all [hkl] reciprocal lattice vectors that lie within
        the Ewald sphere of radius k_mag and satisfy the excitation error cutoff s_max.

        Args:
            ewald (EwaldSphere): An EwaldSphere object which contains the beam energy used.
            max_sg (float): The maximum allowed excitation error.
            max_g (float): The maximum allowed reciprocal lattice vector length. If `None`, no restriction is placed on the maximum.
            kx, ky (float): The beam tilt. Assumed to be small. Default `0`.
            exclude_000 (bool): Whether or not to exclude the [000] reflection. Default `True`.

        Returns:
            ElectronDiffractionSpots:
        """
        max_sg = abs(max_sg)

        recip = self.struct.lattice.reciprocal_lattice_crystallographic
        matrix = recip.matrix * 10 # inv angstrom to inv nm

        # Estimate max h,k,l needed to cover |g| <= 2*k (Ewald sphere diameter), give a little more just to make sure
        if max_g is not None:
            g_max = 2.1 * max_g
        else:
            g_max = 2.1 * ewald.k_z
        max_index = int(np.ceil(g_max / np.min(np.linalg.norm(matrix, axis=1))))

        h, k, l = np.mgrid[-max_index:max_index+1,
                           -max_index:max_index+1,
                           -max_index:max_index+1]
        hkl = np.stack((h, k, l), axis=-1).reshape(-1, 3)  # shape (N,3)

        if exclude_000:
            hkl = hkl[np.any(hkl, axis=1),:] # remove [0,0,0]

        g = hkl @ matrix
        q = np.linalg.norm(g, axis=1)
        
        if max_g is not None:
            mask_max_g = (q <= max_g)

            hkl= hkl[mask_max_g, :]
            g  = g  [mask_max_g, :]
            q  = q  [mask_max_g]
                    
        s = ewald.sg_en_masse(g, kx,ky)
        mask_max_sg = np.abs(s) <= max_sg
        hkl = hkl[mask_max_sg, :]
        g   = g  [mask_max_sg, :]
        q   = q  [mask_max_sg]
        s   = s  [mask_max_sg]

        return ElectronDiffractionSpots(struct=self.struct, hkl=hkl, g=g, q=q, sg=s)
    def powder_hkl(self, ewald:EwaldSphere, max_sg:float, max_g:float=None, num_orientations:int=100, texture:float=0):
        if texture != 0:
            raise ValueError('Texture not yet implemented!')

        handler = StructureHandler( self.struct.copy() )
        for i in range(num_orientations):
            handler.align_structure(np.random.uniform(-1,1, size=3), np.random.uniform(-1,1, size=3))
            if i == 0:
                results = handler.get_excitable_hkl(ewald=ewald, max_sg=max_sg, max_g=max_g)
            else:
                results.extend(handler.get_excitable_hkl(ewald=ewald, max_sg=max_sg, max_g=max_g))
        return results


class ElectronDiffractionSpots:
    def __init__(self, struct:Structure, hkl:np.ndarray, g:np.ndarray=None, q:np.ndarray=None, sg:np.ndarray=None):
        self.hkl = hkl
        if g is None:
            recip = self.struct.lattice.reciprocal_lattice_crystallographic
            g = self.hkl @ recip.matrix * 10 # inv angstrom to inv nm
        self.g = g
        if q is None:
            q = np.linalg.norm(self.g, axis=1)
        self.q = q # lazily evaluated if not provided
        if sg is None:
            sg = np.zeros_like(self.q)
        self.sg = sg

        self.default_electron_atomic_form_factors = default_electron_atomic_form_factors
        self.calculate_structure_factor(struct)
    def extend(self, other):
        # NOTE: the same hkl could end up being defined here, but they could correspond to different sg or Fg, so they should still be kept separate
        self.hkl = np.vstack( (self.hkl, other.hkl) )
        self.g   = np.vstack( (self.g,   other.g)   )
        self.q   = np.hstack( (self.q,   other.q)   )
        self.sg  = np.hstack( (self.sg,  other.sg)  )
        self.Fg  = np.hstack( (self.Fg,  other.Fg)  )
    def calculate_structure_factor(self, struct:Structure):
        """
        Complex F_g for electrons.
        Honors occupancies.
        """
        self.Fg = np.zeros_like(self.q, dtype=np.complex128)
        for site in struct.sites:
            # species can be a Composition; iterate with occupancies
            for sp, occ in site.species.items():
                f_q = self.default_electron_atomic_form_factors[sp.symbol](self.q)

                phase = np.exp(2j*np.pi*np.dot(self.hkl, site.frac_coords))
                self.Fg += occ * f_q * phase
    def kinematical_excitation_err_correction(self, sample_thickness:float):
        '''See Fultz and Howe, chapters 6, 8, and 13. Mostly contained within chapter 8.'''
        return np.sinc(np.pi*self.sg*sample_thickness) * sample_thickness # squared later
    def get_intensity(self, sample_thickness:float=None, normalize:bool=True, eps:float=1e-9):
        """
        Calculate the diffraction intensity for unique in-plane g-vectors.
        This method groups structure factors (Fg) by unique in-plane g-vectors, sums them,
        applies optional excitation error correction for a given sample thickness, and computes
        the intensity as the squared modulus of the summed structure factors. Intensities below
        a specified threshold (eps) are filtered out. Optionally, the intensities can be normalized.
        
        Args:
            sample_thickness (float): The thickness of the sample. If provided and self.sg is not None, applies excitation error correction to the structure factors.
            normalize (bool): If True, normalize the output intensities so that their sum is 1.
            eps (float): Minimum intensity threshold. Intensities below this value are excluded from the output.
        
        Returns:
            hkl (np.ndarray): Miller indices corresponding to the unique g-vectors with intensity above the threshold.
            unique_g (np.ndarray): Unique in-plane g-vectors (shape: [N, 2]) with intensity above the threshold.
            I (np.ndarray): Intensities corresponding to each unique g-vector, optionally normalized.
        """
        # Find unique g vectors (in the plane of the detector) and map each row to a group index, so that I can sum the structure factors before doing abs()^2
        unique_g, idx, inv = np.unique(self.g[:,:2], axis=0, return_index=True, return_inverse=True)
        hkl = self.hkl[np.unique(idx),:] # get associated hkl with these unique_g. This logic won't work for multiple layers as those could overlap in the same spot but originate from different hkl.

        # Sum vals by group index
        Fg = np.zeros(len(unique_g), dtype=complex)
        # Accumulate with np.add.at
        if sample_thickness is not None and self.sg is not None:
            np.add.at(Fg, inv, self.Fg * self.kinematical_excitation_err_correction(sample_thickness=sample_thickness))
        else:
            np.add.at(Fg, inv, self.Fg )

        I = np.abs(Fg)**2

        hkl = hkl[I>eps]
        unique_g = unique_g[I>eps]
        Fg = Fg[I>eps]
        I = I[I>eps]

        if normalize:
            I /= np.sum(I)

        return hkl, unique_g, Fg, I
    def pattern_as_array(self, g_max:float, dq:float, sample_thickness:float=None):
        M = int(2*g_max/dq)
        shape = np.array([M,M])
        arr = np.zeros(shape, dtype=complex)

        _,g,Fg,_ = self.get_intensity(sample_thickness=sample_thickness)

        g_mask = (g[:,0] >= -g_max+dq)&\
                 (g[:,0] <= +g_max-dq)&\
                 (g[:,1] >= -g_max+dq)&\
                 (g[:,1] <= +g_max-dq)

        ij = np.rint(g[g_mask]/dq).astype(int)[:,:2] + shape[None,:]//2

        np.add.at(arr, (ij[:,0], ij[:,1]), Fg[g_mask]) # Add the structure factors that overlap on the same pixel before calculating intensities
        # arr[ij[:,0], ij[:,1]] = I[g_mask] # this assignment will ignore intensities falling on the same pixel; it only uses the last assignment
        return np.abs(arr)**2
    def meshgrid(self, g_max:float, dq:float):
        q = np.linspace(-g_max, g_max, int(2*g_max/dq))
        return np.meshgrid(q,q)
    def pcolor(self, axes:Axes, tight:bool=True, Nq:int=100, sample_thickness:float=None, label_hkl:bool=False, nan_color='black', color='gray', **kwargs):
        g_max = np.max(self.q)
        if tight:
            g_max /= np.sqrt(2)
        
        dq = 2*g_max/Nq

        if not axes.get_xlabel():
            axes.set_xlabel('$nm^{-1}$')
        if not axes.get_ylabel():
            axes.set_ylabel('$nm^{-1}$')
        
        arr = self.pattern_as_array(g_max,dq, sample_thickness=sample_thickness)
        qx,qy = self.meshgrid(g_max,dq)

        if 'norm' in kwargs.keys():
            if kwargs['norm'].lower() == 'log':
                arr[arr==0] = np.nan
        cmap = colormaps[color]
        cmap.set_bad(nan_color)

        # Use a masked array so that nan values are handled by the colormap
        arr_masked = np.ma.masked_invalid(arr)

        im = axes.pcolormesh(qx, qy, arr_masked, cmap=cmap, **kwargs)

        if label_hkl:
            pass

        return im
    def hist(self, axes:Axes, radial_weight:bool=True, sample_thickness:float=None, **kwargs):
        _,g,_,I = self.get_intensity(sample_thickness=sample_thickness)
        if 'weights' not in kwargs.keys():
            w = I
            if radial_weight:
                w /= np.linalg.norm(g,axis=1) # don't use q here because of geometric line broadening
            kwargs.update(weights=w)

        if not axes.get_xlabel():
            axes.set_xlabel('$q (nm^{-1})$')

        return axes.hist(np.linalg.norm(g, axis=1), **kwargs)

