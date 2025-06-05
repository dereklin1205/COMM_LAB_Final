import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import random

#Registers for measurement bases and keys
basesAsja, keyAsja = [],[]
basesBalvis, keyBalvis = [],[]
class E91Protocol:
    def __init__(self, key_length=200):
        """
        Initialize E91 protocol simulation
        
        Args:
            key_length (int): Number of bits in the key
        """
        self.key_length = key_length
        self.basesAsja = []
        self.basesBalvis = []
        self.keyAsja = []
        self.keyBalvis = []
        self.final_key_Asja = []
        self.final_key_Balvis = []
        self.diffAsja = []
        self.diffAsja_bases = []
        self.diffBalvis = []
        self.diffBalvis_bases = []
    def run_protocol(self):
        """
        Run the E91 protocol to generate a shared key
        """
        self.gen_key_and_measure(self.key_length)
        # print(f"Asja's bases: {self.basesAsja}")
        # print(f"Balvis's bases: {self.basesBalvis}")
        # print(f"Asja's key: {self.keyAsja}")
        # print(f"Balvis's key: {self.keyBalvis}")
        self.compare_bases()
        self.compute_CHSH()
        print("E91 protocol completed successfully.")
        print(f"final key is same for Asja and Balvis ? {self.final_key_Asja == self.final_key_Balvis}")
    def eve_intercept(self):
        
    def compare_bases(self):
        for i in range(self.key_length):
            if self.basesAsja[i] == self.basesBalvis[i]:
                self.final_key_Asja.append(self.keyAsja[i])
                self.final_key_Balvis.append(self.keyBalvis[i])
            else:
                self.diffAsja.append(self.keyAsja[i])
                self.diffAsja_bases.append(self.basesAsja[i])
                self.diffBalvis.append(self.keyBalvis[i])
                self.diffBalvis_bases.append(self.basesBalvis[i])
        print(f"Final key for Asja: {self.final_key_Asja}")
        print(f"Final key for Balvis: {self.final_key_Balvis}")
    def compute_CHSH(self):
        ##ZW
        sameZW = 0
        diffZW = 0
        for i,(baseA, baseB) in enumerate(zip(self.diffAsja_bases, self.diffBalvis_bases)):
            if baseA == 'Z' and baseB == 'W':
                if self.diffAsja[i] == self.diffBalvis[i]:
                    sameZW += 1
                else:
                    diffZW += 1
        totalZW=sameZW+diffZW
        if totalZW!=0:
            ZW=(sameZW-diffZW)/totalZW
        else:
            ZW=0
        ##ZV
        sameZV = 0
        diffZV = 0
        for i,(baseA, baseB) in enumerate(zip(self.diffAsja_bases, self.diffBalvis_bases)):
            if baseA == 'Z' and baseB == 'V':
                if self.diffAsja[i] == self.diffBalvis[i]:
                    sameZV += 1
                else:
                    diffZV += 1
        totalZV=sameZV+diffZV
        if totalZV!=0:
            ZV=(sameZV-diffZV)/totalZV
        else:
            ZV=0
        ##XW
        sameXW = 0
        diffXW = 0
        for i,(baseA, baseB) in enumerate(zip(self.diffAsja_bases, self.diffBalvis_bases)):
            if baseA == 'X' and baseB == 'W':
                if self.diffAsja[i] == self.diffBalvis[i]:
                    sameXW += 1
                else:
                    diffXW += 1
        totalXW=sameXW+diffXW
        if totalXW!=0:
            XW=(sameXW-diffXW)/totalXW
        else:
            XW=0
        ##XV
        sameXV = 0
        diffXV = 0
        for i,(baseA, baseB) in enumerate(zip(self.diffAsja_bases, self.diffBalvis_bases)):
            if baseA == 'X' and baseB == 'V':
                if self.diffAsja[i] == self.diffBalvis[i]:
                    sameXV += 1
                else:
                    diffXV += 1
        totalXV=sameXV+diffXV
        if totalXV!=0:
            XV=(sameXV-diffXV)/totalXV
        else:
            XV=0
        S = ZW + ZV + XW - XV
        print(f"CHSH value: {S}")
        print(f"CHSH value is {'greater' if S > 2 else 'less or equal'} than 2, {'indicating potential eavesdropping' if S > 2 else 'indicating no eavesdropping detected'}.")
    def gen_key_and_measure(self, key_length=200):
        for i in range(key_length):  #Asja prepares 200 EPR pairs
            qc = QuantumCircuit(2, 2)

            #Creating entanglement
            qc.h(0)
            qc.cx(0, 1)

            #Asja chooses measurement basis
            #random select 0,1,2
            choiceAsja = random.randint(0, 2)
            choiceBalvis = random.randint(0, 2)
            if choiceAsja==0:#measurement in Z basis
                self.basesAsja.append('Z')
            if choiceAsja==1:#measurement in X basis
                qc.h(0)
                self.basesAsja.append('X')
            if choiceAsja==2:#measurement in W basis
                qc.s(0)
                qc.h(0)
                qc.t(0)
                qc.h(0)
                self.basesAsja.append('W')
            
            #Balvis chooses measurement basis
            ##random selecy 0,1,2
            
            if choiceBalvis==0:#measurement in Z basis
                self.basesBalvis.append('Z')
            if choiceBalvis==1:#measurement in W basis
                qc.s(1)
                qc.h(1)
                qc.t(1)
                qc.h(1)
                self.basesBalvis.append('W')
            if choiceBalvis==2:#measurement in V basis
                qc.s(1)
                qc.h(1)
                qc.tdg(1)
                qc.h(1)
                self.basesBalvis.append('V')

            #Measurement
            qc.measure([0, 1], [0, 1])
            #Simulating the circuit
            simulator = AerSimulator()
            job = simulator.run(qc, shots=100)
            result = job.result()
            counts = result.get_counts(qc)
            # print(f"Measurement results: {counts}")
            #Saving results
            result=list(counts.keys())[0] #retrieve key from dictionary
            self.keyAsja.append(int(result[0])) #saving first qubit value in Asja's key register 
            self.keyBalvis.append(int(result[1])) #and second to Balvis
if __name__ == "__main__":
    e91 = E91Protocol(key_length=1000)
    e91.run_protocol()
    # print(f"Final key for Asja: {e91.final_key_Asja}")
    # print(f"Final key for Balvis: {e91.final_key_Balvis}")
    # print(f"Difference in keys for Asja: {e91.diffAsja}")
    # print(f"Difference in bases for Asja: {e91.diffAsja_bases}")
    # print(f"Difference in keys for Balvis: {e91.diffBalvis}")
    # print(f"Difference in bases for Balvis: {e91.diffBalvis_bases}")