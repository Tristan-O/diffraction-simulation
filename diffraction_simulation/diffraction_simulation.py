import numpy as np
import pandas as pd
import os
from collections import defaultdict
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.ext.matproj import MPRester


default_form_factors_path = './2025-08-12_Atomic-Form-Factors-0-20inversenm.csv'
if not os.path.exists(default_form_factors_path):
    from urllib.request import Request, urlopen
    from bs4 import BeautifulSoup

    url = "https://it.iucr.org/Cb/ch4o3v0001/sec4o3o2/"

    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(req).read()

    soup = BeautifulSoup(webpage, 'html.parser')

    # Find all tables
    tables = soup.find_all('table')

    table = str(tables[5]).replace('−', '-') # unicode minus sign
    form_factors_df = pd.read_html(table, skiprows=4)[0]
    form_factors_df = form_factors_df.dropna()
    print(f"Table {i}:")
    form_factors_df.columns = ['Element', 'Z'] + [f'a{i}' for i in range(1,6)] + [f'b{i}' for i in range(1,6)]
    form_factors_df['source'] = url
    print(form_factors_df.iloc[0])
    print(len(form_factors_df['Element']))
    # raise ValueError('Do not scrape again!')
    form_factors_df.to_csv(default_form_factors_path, index=False)
form_factors_df = pd.read_csv(default_form_factors_path)

class ElectronAtomicFormFactor:
    @staticmethod
    def get_element_info(element:str|int|float):
        if isinstance(element,(int,float)):
            row = form_factors_df.loc[form_factors_df['Z'] == element]
        else:
            row = form_factors_df.loc[form_factors_df['Element'] == element]

        if row.empty:
            raise ValueError('Provided element not found!')
        else:
            return {k:list(e.values())[0] for k,e in row.to_dict().items()}
    @classmethod
    def from_default(cls, element:str|int|float):
        info = cls.get_element_info(element)
        A = np.array( [info[f'a{i}'] for i in range(1,6)] )
        B = np.array( [info[f'b{i}'] for i in range(1,6)] )/100 # from my list, B is in A^-2, but I want to work wth q in nm^-1, so I divide by 100
        def func(q:float|np.ndarray, A=A, B=B):
            return np.inner(A, np.exp(-np.outer(np.square(q),B)))
        return cls(info['Element'], func)
    def __init__(self, name, func):
        self.name = name
        self.func = func
    def __call__(self, q:float|np.ndarray):
        return self.func(q)

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
    COMMON_MATERIAL_IDS = dict(Si_Fd3m='mp-149', # diamond cubic Si
                              Se_Pm3m='mp-7755', # simple cubic Se, the only known single-element simple cubic structure
                              HfO2_Pca21='mp-685097', # ferroelectric orthorhombic HfO2
                              HfO2_P21c='mp-352', # monoclinic HfO2
                              HfO2_P42nmc='mp-1018721') # tetragonal HfO2
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
    def get_excitable_hkl(self, ewald:EwaldSphere, max_sg:float, max_g:float=None, kx:float=0, ky:float=0):
        """
        Returns all [hkl] reciprocal lattice vectors that lie within
        the Ewald sphere of radius k_mag and satisfy the excitation error cutoff s_max.

        Args:
            structure (Structure): pymatgen Structure (lattice already aligned if needed)

        Returns:
            list[ElectronDiffractionSpot]
        """
        max_sg = abs(max_sg)

        recip = self.struct.lattice.reciprocal_lattice_crystallographic
        a_vec, b_vec, c_vec = recip.matrix * 10 # inv angstrom to inv nm

        # Estimate max h,k,l needed to cover |g| <= 2*k (Ewald sphere diameter), give a little more just to make sure
        if max_g is not None:
            g_max = 2.1 * max_g
        else:
            g_max = 2.1 * ewald.k_z
        max_index = int(np.ceil(g_max / np.min(np.linalg.norm([a_vec, b_vec, c_vec], axis=1))))

        h, k, l = np.mgrid[-max_index:max_index+1,
                   -max_index:max_index+1,
                   -max_index:max_index+1]
        hkl = np.stack((h, k, l), axis=-1).reshape(-1, 3)  # shape (N,3)

        g = hkl @ recip.matrix * 10 # inv angstrom to inv nm
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
            handler.align_structure(np.random.uniform(0,1, size=3), np.random.uniform(0,1, size=3))
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
        self.Fg = self.calculate_structure_factor(struct)
    def extend(self, other):
        # print(self.hkl.shape, other.hkl.shape)
        self.hkl = np.vstack((self.hkl, other.hkl))
        # print(self.hkl.shape)
        self.g = np.vstack((self.g, other.g))
        self.q = np.hstack((self.q, other.q))
        self.sg = np.hstack((self.sg, other.sg))
        self.Fg = np.hstack((self.Fg, other.Fg))
    def calculate_structure_factor(self, struct:Structure):
        """
        Complex F_g for electrons.
        Uses site property 'B' for the Debye Waller factor (Å^2) if present. 
        Honors occupancies.
        """

        Fg = np.zeros_like(self.q, dtype=np.complex128)
        for site in struct.sites:
            # species can be a Composition; iterate with occupancies
            for sp, occ in site.species.items():
                f_q = ElectronAtomicFormFactor.from_default(sp.symbol)(self.q)
                if len(f_q) == 1:
                    f_q = f_q[0]
                # Debye–Waller if available on the site, else 0
                B = site.properties.get("B", 0.0)
                f_q *= np.exp(-B * self.q**2 / (4*np.pi**2))
                phase = np.exp(2j*np.pi*np.dot(self.hkl, site.frac_coords))
                Fg += occ * f_q * phase
        return Fg
    def kinematical_excitation_err_correction(self, sample_thickness:float):
        '''See Fultz and Howe, chapters 6, 8, and 13. Mostly contained within chapter 8.'''
        return np.sinc(np.pi*self.sg*sample_thickness)**2 * sample_thickness**2
    def get_intensity(self, sample_thickness:float=None):
        I = np.abs(self.Fg)**2
        if sample_thickness is not None and self.sg is not None:
            I *= self.kinematical_excitation_err_correction(sample_thickness=sample_thickness)
        return I
    def pattern_as_array(self, g_max:float, dq:float, sample_thickness:float=None):
        arr = np.zeros([int(2*g_max/dq)]*2)
        ijk = np.rint(self.g/dq).astype(int)[:,:2] + np.array(arr.shape)[None,:]//2
        arr[ijk[:,0], ijk[:,1]] = self.get_intensity(sample_thickness=sample_thickness)        
        return arr
    def meshgrid(self, g_max:float, dq:float):
        q = np.linspace(-g_max, g_max, int(2*g_max/dq))
        return np.meshgrid(q,q)


if __name__ == '__main__':
    # print(ElectronAtomicFormFactor.from_default('O')([0,10,20]))
    ewald = EwaldSphere(300e3)
    struct = StructureHandler.from_matproj("HfO2_Pca21")

    g_max = 5
    dq = 0.1
    sg_max = 1
    dp = struct.get_excitable_hkl(ewald, sg_max, g_max)
    print('# spots:',len(dp.q))

    from matplotlib import pyplot as plt
    fig,(ax1,ax2) = plt.subplots(1,2, figsize=(10,6))
    # plt.pcolor(*dp.meshgrid(g_max,dq), dp.pattern_as_array(g_max,dq), norm='log')
    # plt.show()

    powder = struct.powder_hkl(ewald, sg_max, g_max, num_orientations=10000)
    arr = powder.pattern_as_array(g_max,dq, sample_thickness=10)
    arr[arr == 0] = np.min(arr[arr!=0])

    ax1.pcolor(*powder.meshgrid(g_max,dq), arr, norm='log')
    ax1.set_xlabel('$nm^{-1}$')
    # plt.show()

    ax2.hist(powder.q, weights=powder.get_intensity(10)/powder.q, bins=100)
    ax2.set_xlim(1,g_max)
    plt.show()