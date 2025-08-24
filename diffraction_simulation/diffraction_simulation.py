from __future__ import annotations
from collections import defaultdict
import numpy as np
from matplotlib.axes import Axes
from matplotlib import colormaps
from matplotlib import pyplot as plt
from pymatgen.core import Structure
from pymatgen.core.operations import SymmOp
from pymatgen.transformations.standard_transformations import RotationTransformation
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.ext.matproj import MPRester
from .electron_atomic_form_factor import default_electron_atomic_form_factors
from .misc import convert_unit, pretty_unit

class EwaldSphere:
    hbar_c = 197 # eV-nm
    m0 = 511e3 # eV
    def __init__(self, energy_eV:float):
        self.energy_eV = energy_eV
    @property
    def wavelength(self):
        '''Get the wavelength in the length units of StructureHandler (default nm)'''
        return convert_unit(2*np.pi*self.hbar_c / np.sqrt(self.energy_eV**2 + 2*self.m0*self.energy_eV), 'nm', StructureHandler.LENGTH_UNIT)
    @property
    def k_z(self):
        '''Get the wavevector in cyc/wavelength units (default cyc/nm), crystallographic convention'''
        return 1/self.wavelength
    @property
    def k_z_angular(self):
        '''Get the wavevector in rad/wavelength unit (default rad/nm), physics convention'''
        return 2*np.pi*self.k_z
    def sg(self, g:tuple[float,float,float], kx:np.ndarray=0, ky:np.ndarray=0):
        '''Returns the excitation error of a given point in reciprocal space, in cyc/wavelength unit'''
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
    LENGTH_UNIT = 'nm' # Any length unit. Acceptable strings include 'nm', 'angstrom', etc.
    MATERIALS_PROJECT_API = 'E3sCeYTbOOxpD9gtJ1EgevVyhHUn3w2J'
    COMMON_MATERIAL_IDS = dict(Si_Fd3m      = 'mp-149',     # diamond cubic Si
                               Se_Pm3m      = 'mp-7755',    # simple cubic Se, the only known single-element simple cubic structure
                               HfO2_Pca21   = 'mp-685097',  # ferroelectric orthorhombic HfO2
                               HfO2_P21c    = 'mp-352',     # monoclinic HfO2
                               HfO2_P42nmc  = 'mp-1018721', # tetragonal HfO2
                               TiN_Fm3m     = 'mp-492')     # cubic TiN
    @staticmethod
    def _uniformly_sampled_points_on_sphere(num_points:int=100):
        """
        Generates uniformly distributed points on the surface of a sphere using the Fibonacci lattice method.

        Args:
            num_points (int, optional): The number of points to sample on the sphere. Defaults to 100.

        Returns:
            tuple of np.ndarray: A tuple (theta, phi) where:
                - theta (np.ndarray): Array of polar angles (in radians), shape (num_points,).
                - phi (np.ndarray): Array of azimuthal angles (in radians), shape (num_points,).

        Notes:
            - The Fibonacci lattice method provides an efficient way to distribute points APPROXIMATELY uniformly on a sphere.
            - The returned angles (theta, phi) can be used to convert to Cartesian coordinates for 3D visualization or further computation.
            - Theta is the polar angle measured from the positive z-axis (0 <= theta <= pi).
            - Phi is the azimuthal angle in the x-y plane from the positive x-axis (0 <= phi < 2*pi).
        """
        # Fibonacci lattice method for uniform sampling on a sphere
        indices = np.arange(0, num_points, dtype=float) + 0.5
        phi = (2 * np.pi * indices / ((1 + np.sqrt(5)) / 2)) % (2*np.pi) # golden ratio is the most irrational number
        theta = np.arccos(1 - 2 * indices / num_points)
        return theta, phi
    @staticmethod
    def _euler_rotation_matrix_zxz(alpha:float, beta:float, gamma:float):
        # Euler ZXZ rotation matrix
        ca, cb, cg = np.cos([alpha, beta, gamma])
        sa, sb, sg = np.sin([alpha, beta, gamma])

        Rz1 = np.array([
            [ca, -sa, 0],
            [sa,  ca, 0],
            [ 0,   0, 1]
        ])
        Rx  = np.array([
            [1,   0,    0],
            [0,  cb, -sb],
            [0,  sb,  cb]
        ])
        Rz2 = np.array([
            [cg, -sg, 0],
            [sg,  cg, 0],
            [ 0,   0, 1]
        ])
        return Rz1 @ Rx @ Rz2
    @staticmethod
    def Rx(theta:float):
        Rx  = np.array([
            [1, 0,             0            ],
            [0, np.cos(theta),-np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
        return Rx
    @staticmethod
    def Rz(theta:float):
        Rx  = np.array([
            [np.cos(theta),-np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0,             0,             1],
        ])
        return Rx
    @staticmethod
    def _uniformly_sampled_euler_rotation_matrices_zxz(num_points_sqrt:int=100):
        alpha, beta = StructureHandler._uniformly_sampled_points_on_sphere(num_points=num_points_sqrt)
        gamma = np.linspace(0,2*np.pi, num_points_sqrt)

        alpha = np.stack([alpha]*num_points_sqrt).T.reshape(-1)
        beta  = np.stack([beta ]*num_points_sqrt).T.reshape(-1)
        gamma = np.stack([gamma]*num_points_sqrt).reshape(-1)

        ca, cb, cg = np.cos([alpha, beta, gamma])
        sa, sb, sg = np.sin([alpha, beta, gamma])
        zeros = np.zeros_like(alpha)
        ones  = np.ones_like(alpha)

        Rz1 = np.array([
            [ca,     -sa,  zeros],
            [sa,     ca,   zeros],
            [zeros, zeros, ones]
        ])
        Rx  = np.array([
            [ones,  zeros, zeros],
            [zeros, cb,    -sb],
            [zeros, sb,    cb]
        ])
        Rz2 = np.array([
            [cg,    -sg,   zeros],
            [sg,     cg,   zeros],
            [zeros, zeros, ones]
        ])
        # all matrices are shape 3x3xN here

        res = np.einsum('ijk,jlk->ilk', Rz1, Rx)
        res = np.einsum('ijk,jlk->kil', res, Rz2)
        # res now has shape (N,3,3)
        
        return res
    @classmethod
    def from_matproj(cls, material_id:str="Si_Fd3m", conventional_unit_cell:bool=True, shift_basis:bool=True):
        if material_id in cls.COMMON_MATERIAL_IDS.keys():
            material_id = cls.COMMON_MATERIAL_IDS[material_id]
        with MPRester(cls.MATERIALS_PROJECT_API) as mpr:
            struct = mpr.get_structure_by_material_id(material_id=material_id, 
                                                        conventional_unit_cell=conventional_unit_cell)
        
        res = cls(struct)
        if shift_basis:
            res.shift_to_origin()
        return res
    def __init__(self, struct:Structure):
        '''Here surface normal are the [uvw] (fractional coordinates) to use for aligning the Structure's surface normal to the incident beam.
           This is 
        '''
        self.struct = struct
    def shift_to_origin(self, element:str=None):
        """
        Translate the structure so that the atom of the selected element (default, the highest Z) closest to the origin is moved to the origin.
        """
        if element is None:
            # Find heaviest element in the structure
            heaviest_Z = -1
            for site in self.struct:
                if site.specie.Z > heaviest_Z:
                    heaviest_Z = site.specie.Z
                    # print(site.specie)
                    element = str(site.specie)
        
        # Find the coord closest to the origin
        coords = []
        for i, site in enumerate(self.struct):
            if str(site.specie) == element:
                coords.append(site.coords)
        i = np.argmin(np.linalg.norm(coords, axis=1))
        shift = coords[i]
        
        self.struct = self.struct.apply_operation(SymmOp.from_rotation_and_translation(np.eye(3), -shift))

        return self
    def align_structure_uvw_to_z(self, 
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
    def apply_matrix_transformation(self, R:np.ndarray):
        """
        Apply a rotation matrix R (3x3) to the lattice vectors of the structure.
        The atomic fractional coordinates remain unchanged.
        """
        lat_matrix = np.array(self.struct.lattice.matrix)
        new_lattice_matrix = lat_matrix @ R # pymatgen puts lattice vectors as rows
        self.struct = Structure(
            lattice=new_lattice_matrix,
            species=self.struct.species,
            coords=self.struct.frac_coords,
            coords_are_cartesian=False
        )
        return self
    def BCT2FCC_transform(self):
        '''
        Transforms lattice vectors a1 and a2 to (a1-a2) and (a1+a2). 
        Then realigns those directions with the x and y cartesian axes.
        The effect is to take a BCT structure and turn it into a (nearly) FCC structure.
        (It is only truly FCC if the length of a3 was such that it now has the same length as a1' and a2'.)
        There is no strain applied.
        '''
        # define the transformation matrix
        M = [[1, -1, 0],
             [1,  1, 0],
             [0,  0, 1]]

        # apply the transformation
        self.struct.make_supercell(M)
        self.struct = RotationTransformation([0,0,1], 45).apply_transformation(self.struct)
        # self.struct.apply_operation(SymmOp.from_rotation_and_translation(R,[0,0,0]))
        return self
    def _get_atom_positions(self, low_frac:tuple[float,float,float]=(-0.1,-0.1,-0.1),
                                  high_frac:tuple[float,float,float]=(1.1,1.1,1.1))->dict[str,list[np.ndarray]]:
        a_lo = low_frac[0]
        b_lo = low_frac[1]
        c_lo = low_frac[2]
        a_hi = high_frac[0]
        b_hi = high_frac[1]
        c_hi = high_frac[2]

        # pad a unit cell in all three directions
        supercell = self.struct.make_supercell(3,in_place=False)
        supercell.translate_sites(range(len(supercell)), [-1./3., -1./3., -1./3.], frac_coords=True, to_unit_cell=False)

        # Get the atom coordinates (cartesian)
        coords = defaultdict(list)
        for el in set(self.struct.species):
            for site in supercell:
                xf,yf,zf = self.struct.lattice.get_fractional_coords(site.coords) # go through sites in supercell, map them back to the single cell, keep only the atoms whose coords fall within the desired unit cell

                if (  a_lo<=xf<=a_hi and 
                      b_lo<=yf<=b_hi and
                      c_lo<=zf<=c_hi and 
                      el == site.specie  ):
                    coords[str(site.specie)].append( convert_unit(site.coords, 'angstrom', StructureHandler.LENGTH_UNIT) ) # angstrom to nm
        return coords
    def plot_unit_cell_2d(self, axes:Axes=None, 
                       low_frac:tuple[float,float,float]=(-0.1,-0.1,-0.1),
                       high_frac:tuple[float,float,float]=(1.1,1.1,1.1),
                       origin:tuple=(0,0,0),
                       proj_ax_uvw:tuple[float,float,float]=(0,0,1),
                       x_ax_uvw:tuple[float,float,float]=(1,0,0)):
        '''Plot the unit cell, as viewed along the desired axis.'''

        normal = self.struct.lattice.matrix @ np.array(proj_ax_uvw) 
        normal /= np.linalg.norm(normal)

        ref = self.struct.lattice.matrix @ np.array(x_ax_uvw) 
        ref /= np.linalg.norm(ref)

        # First basis vector in plane (orthogonal to n)
        u = ref - np.dot(ref, normal) * normal
        u /= np.linalg.norm(u)
        
        # Second basis vector (cross product ensures orthogonality)
        v = np.cross(normal, u)
        v /= np.linalg.norm(v)
        

        for el, coords in self._get_atom_positions(low_frac=low_frac, high_frac=high_frac).items():
            # Project points into 2D coordinates
            coords = coords - np.array(origin)
            x_coords = np.dot(coords, u)
            y_coords = np.dot(coords, v)
            coords_2d = np.vstack([x_coords, y_coords]).T
            x,y = np.transpose(coords_2d)
            axes.scatter(x, y, label=el)
        return axes
    def plot_unit_cell_3d(self, axes:Axes=None, 
                          low_frac:tuple[float,float,float]=(-0.1,-0.1,-0.1),
                          high_frac:tuple[float,float,float]=(1.1,1.1,1.1),
                          origin:tuple=(0,0,0),
                          frame:bool=True, colors:dict=None, sizes:dict={'default':100}):
        '''Implemented before I knew about https://pymatgen.org/pymatgen.vis.html#pymatgen.vis.structure_chemview.quick_view'''
        if axes is None:
            fig = plt.figure()
            axes = fig.add_subplot(111, projection='3d', proj_type='ortho')
        axes.set_title(f'${self.struct.get_space_group_info()[0]}$')
        axes.set_aspect('equal')
        axes.set_xlabel('x')
        axes.set_ylabel('y')
        axes.set_zlabel('z')

        for el, coords in self._get_atom_positions(low_frac=low_frac, high_frac=high_frac).items():
            x,y,z = np.transpose(coords)
            axes.scatter(x-origin[0],
                         y-origin[1],
                         z-origin[2], 
                         label=el,
                         s=sizes[el] if el in sizes else sizes['default'],
                         color=colors[el] if colors is not None else None)
        
        if frame:
            o = [0,0,0]
            a1,a2,a3 = convert_unit(self.struct.lattice.matrix, 'angstrom', StructureHandler.LENGTH_UNIT) # angstrom to nm
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
                axes.plot(xs, ys, zs, color='black')

        return axes
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
        matrix = convert_unit(recip.matrix, '1/angstrom', '1/'+StructureHandler.LENGTH_UNIT) # inv angstrom to inv nm

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
        results = None

        num = int(np.sqrt(num_orientations))
        theta,phi = self._uniformly_sampled_points_on_sphere(num)
        
        for th,ph in zip(theta,phi):
            R1 = StructureHandler.Rz(ph) @ StructureHandler.Rx(th) # rotate about z by 0<phi<2pi (azimuthal), about x by 0<theta<pi (polar)
            handler.apply_matrix_transformation(R1)

            R2 = StructureHandler.Rz(2*np.pi/num)
            for _ in range(num):
                handler.apply_matrix_transformation(R2) # doing this effectively indexes from 1, already rotated at i=0
                if results is None:
                    results = handler.get_excitable_hkl(ewald=ewald, max_sg=max_sg, max_g=max_g)
                else:
                    results.extend(handler.get_excitable_hkl(ewald=ewald, max_sg=max_sg, max_g=max_g))
            handler.apply_matrix_transformation(R2@np.linalg.inv(R1))
        return results


class ElectronDiffractionSpots:
    def __init__(self, struct:Structure, hkl:np.ndarray, g:np.ndarray=None, q:np.ndarray=None, sg:np.ndarray=None):
        self.hkl = hkl
        if g is None:
            recip = self.struct.lattice.reciprocal_lattice_crystallographic
            g = convert_unit(self.hkl @ recip.matrix, '1/angstrom', '1/'+StructureHandler.LENGTH_UNIT) # inv angstrom to inv nm
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
                f_q = self.default_electron_atomic_form_factors[sp.symbol](self.q, q_unit='1/' + StructureHandler.LENGTH_UNIT)

                phase = np.exp(2j*np.pi*np.dot(self.hkl, site.frac_coords))
                self.Fg += occ * f_q * phase
    def kinematical_excitation_err_correction(self, sample_thickness:float):
        '''See Fultz and Howe, chapters 6, 8, and 13. Mostly contained within chapter 8.'''
        return np.sinc(np.pi*self.sg*sample_thickness) * sample_thickness # squared later TODO address sample_thickness units
    def get_intensity(self, sample_thickness:float=None, eps:float=1e-9):
        """
        Calculate the diffraction intensity for unique in-plane g-vectors.
        This method groups structure factors (Fg) by unique in-plane g-vectors, sums them,
        applies optional excitation error correction for a given sample thickness, and computes
        the intensity as the squared modulus of the summed structure factors. Intensities below
        a specified threshold (eps) are filtered out. Optionally, the intensities can be normalized.
        
        Args:
            sample_thickness (float): The thickness of the sample. If provided and self.sg is not None, applies excitation error correction to the structure factors.
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

        # if normalize: # this is a nonsensical way of normalizing
        #     I /= np.sum(I)

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
            axes.set_xlabel(f'${pretty_unit(StructureHandler.LENGTH_UNIT)}^{{-1}}$')
        if not axes.get_ylabel():
            axes.set_ylabel(f'${pretty_unit(StructureHandler.LENGTH_UNIT)}^{{-1}}$')
        
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
            axes.set_xlabel(f'$q ({pretty_unit(StructureHandler.LENGTH_UNIT)}^{{-1}})$')

        return axes.hist(np.linalg.norm(g, axis=1), **kwargs)
    @staticmethod
    def stacked_hist(axes:Axes, *dps:ElectronDiffractionSpots, radial_weight:bool=True, sample_thickness:float=None, **kwargs):
        all_g,all_w = [],[]
        for dp in dps:
            _,g,_,I = dp.get_intensity(sample_thickness=sample_thickness)
            if 'weights' not in kwargs.keys():
                w = I
                if radial_weight:
                    w /= np.linalg.norm(g,axis=1) # don't use q here because of geometric line broadening
                all_w.append(w)

            all_g.append(np.linalg.norm(g[:,:2], axis=1))

        if not axes.get_xlabel():
            axes.set_xlabel(f'$q ({pretty_unit(StructureHandler.LENGTH_UNIT)}^{{-1}})$')
        
        return axes.hist(all_g, weights=all_w, stacked=True, **kwargs)
    def hist2d(self, axes:Axes, sample_thickness:float=None, nan_color:str='black', cmap:str='gray', **kwargs):
        _,g,_,I = self.get_intensity(sample_thickness=sample_thickness)
        if 'weights' not in kwargs.keys():
            kwargs.update(weights=I)

        if not axes.get_xlabel():
            axes.set_xlabel(f'$q_x ({pretty_unit(StructureHandler.LENGTH_UNIT)}^{{-1}})$')
        if not axes.get_ylabel():
            axes.set_ylabel(f'$q_y ({pretty_unit(StructureHandler.LENGTH_UNIT)}^{{-1}})$')

        cmap = colormaps[cmap]
        cmap.set_bad(nan_color)
        return axes.hist2d(g[:,0], g[:,1], cmap=cmap, **kwargs)
