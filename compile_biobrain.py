import numpy as np
import pickle
import time
from brain_architecture import BioBrain # On importe ton architecture

class BioCompiler:
    def __init__(self, brain):
        self.brain = brain
        self.input_size = 100
        self.library = {}
        self.generate_primordial_patterns()

    def generate_primordial_patterns(self):
        """ Génère les signatures électriques uniques pour les 8 concepts """
        concepts = ["SOI", "AUTRUI", "DANGER", "STABILITE", "ECOUTE", "PAROLE", "MOUVEMENT", "LOI"]
        for c in concepts:
            # Chaque concept est un vecteur unique de 100 spikes
            # On utilise une graine fixe pour que l'éducation soit reproductible
            np.random.seed(concepts.index(c)) 
            pattern = (np.random.rand(self.input_size) > 0.8).astype(np.float32)
            self.library[c] = pattern
        print(f"--- Bibliothèque de {len(concepts)} concepts primordiaux générée ---")

    def train_concept(self, name, iterations=1000, target_action=None):
        """ Injecte un concept et renforce les circuits associés """
        pattern = self.library[name]
        
        for i in range(iterations):
            # 1. Injection sensorielle
            s_out, a_out, e_out = self.brain.forward(pattern)
            
            # 2. Si c'est un apprentissage dirigé (ex: PAROLE -> COMMUNIQUER)
            if target_action is not None:
                # On excite artificiellement le pool de neurones de l'action
                start = target_action * self.brain.executive.pool_size
                end = start + self.brain.executive.pool_size
                self.brain.executive.potential[start:end] += 0.5
                self.brain.executive.apply_reinforcement(0.1) # Dopamine
            
            # 3. Application de la plasticité sur toutes les couches
            self.brain.sensory.apply_stdp()
            self.brain.associative.apply_stdp()
            # L'exécutif apprend avec dopamine
            self.brain.executive.apply_stdp()

    def imprint_ethics(self, iterations=2000):
        """ Phase critique : Apprendre au bouclier Limbique à bloquer le danger """
        print("Imprégnation des Lois (Limbic Shielding)...")
        # On combine le pattern AUTRUI et DANGER
        forbidden_pattern = np.clip(self.library["AUTRUI"] + self.library["DANGER"], 0, 1)
        
        for _ in range(iterations):
            # On force le système limbique à réagir violemment à ce pattern
            self.brain.forward(forbidden_pattern)
            self.brain.limbic.potential += 1.0 # On force la douleur
            self.brain.limbic.apply_stdp(learning_rate=0.05) # Plasticité forte
            # On punit l'exécutif s'il essaie de bouger pendant ce pattern
            self.brain.executive.apply_reinforcement(-1.0)

    def compile(self, filename="stable_brain_v1.bio"):
        """ Lance le cycle complet d'éducation et sauvegarde """
        start_time = time.time()
        
        print("Étape 1 : Stabilisation du SOI...")
        self.train_concept("SOI", iterations=1000)
        
        print("Étape 2 : Reconnaissance d'AUTRUI...")
        self.train_concept("AUTRUI", iterations=1000)
        
        print("Étape 3 : Apprentissage du DANGER...")
        self.train_concept("DANGER", iterations=1000)
        
        print("Étape 4 : Imprégnation éthique (LOIS)...")
        self.imprint_ethics()
        
        print("Étape 5 : Apprentissage linguistique (ECOUTE/PAROLE)...")
        self.train_concept("ECOUTE", iterations=1000)
        self.train_concept("PAROLE", iterations=1000, target_action=2) # Action "COMMUNIQUER"
        
        print(f"--- Compilation terminée en {time.time() - start_time:.2f}s ---")
        
        # Sauvegarde
        with open(filename, 'wb') as f:
            pickle.dump({
                'weights_s': self.brain.sensory.weights,
                'weights_a': self.brain.associative.weights,
                'weights_e': self.brain.executive.weights,
                'weights_l': self.brain.limbic.weights,
                'vocabulary': self.library
            }, f)
        print(f"Cerveau sauvegardé sous : {filename}")

    def train_broca_sequence(self, concept_name, word_output, word_index=None):
        """Associate an abstract concept with a Broca output neuron cluster."""
        print(f"Imprégnation linguistique : {concept_name} -> '{word_output}'")
        pattern = self.library[concept_name]

        if word_index is None:
            word_index = abs(hash(word_output)) % self.brain.broca.weights.shape[0]

        for _ in range(500):
            self.brain.forward(pattern)
            self.brain.broca.potential[word_index] += 0.6
            self.brain.broca.apply_stdp()

if __name__ == "__main__":
    # Initialisation d'un cerveau vierge
    my_brain = BioBrain()
    
    # Compilation / Éducation
    compiler = BioCompiler(my_brain)
    compiler.compile()