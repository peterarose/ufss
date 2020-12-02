import os
from ufss.HLG import PolymerVibrations,DiagonalizeHamiltonian,RedfieldConstructor,DiagonalizeLiouvillian,OpenPolymerVibrations

class run:

    def __init__(self,folder,*,for_RKE = False,conserve_memory = False):
        self.for_RKE = for_RKE
        self.conserve_memory = conserve_memory
        self.folder = folder
        try:
            convert(folder)
        except FileNotFoundError:
            pass

        try:
            pv = PolymerVibrations(os.path.join(folder,'params.yaml'))
        except FileNotFoundError:
            raise Exception('Could not find params.yaml file in specified folder')

        if 'bath' in pv.params.keys():
            self.open = True
            if 'site_bath' in pv.params['bath'].keys():
                self.Redfield = True
            else:
                self.Redfield = False
                try:
                    self.fdb_flag = pv.params['bath']['force_detailed_balance']
                except KeyError:
                    self.fdb_flag = False
        else:
            self.open = False

        if self.open:
            self.run_open()
        else:
            self.run_closed()

    def run_closed(self):
        if self.for_RKE:
            pass
        else:
            dh = DiagonalizeHamiltonian(self.folder)

    def run_open(self):
        if self.Redfield:
            dh = DiagonalizeHamiltonian(self.folder)
            rf = RedfieldConstructor(self.folder,conserve_memory=self.conserve_memory)
            dl = DiagonalizeLiouvillian(self.folder,conserve_memory=self.conserve_memory)
        else:
            opv = OpenPolymerVibrations(os.path.join(self.folder,'params.yaml'),for_RKE=self.for_RKE,force_detailed_balance=self.fdb_flag)
