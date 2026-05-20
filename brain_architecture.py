import numpy as np

class BioTensor:
    def __init__(self, shape, name="Cortex", threshold=1.0, decay=0.2):
        self.name = name
        # Initialisation des poids : (sorties, entrées)
        self.weights = np.random.randn(*shape).astype(np.float32) * 0.05
        self.potential = np.zeros(shape[0], dtype=np.float32)
        self.spikes = np.zeros(shape[0], dtype=np.bool_)
        
        # --- Paramètres Physiques ---
        self.base_threshold = threshold
        self.threshold = np.full(shape[0], threshold, dtype=np.float32) # Seuil adaptatif
        self.decay = decay
        self.fatigue = np.zeros(shape[0], dtype=np.float32) # Fatigue neuronale
        
        # --- Traces de Plasticité (STDP) ---
        self.trace_pre = np.zeros(shape[1], dtype=np.float32)
        self.trace_post = np.zeros(shape[0], dtype=np.float32)
        self.trace_decay = 0.9
        
        self.timestamp = 0

    def forward(self, x):
        # 1. Mise à jour de la trace d'entrée (mémoire sensorielle)
        self.trace_pre = (self.trace_pre * self.trace_decay) + x
        
        # 2. Intégration du signal avec Fatigue
        self.potential *= (1.0 - self.decay)
        # On soustrait la fatigue au potentiel entrant
        effective_input = np.dot(self.weights, x) - (self.fatigue * 3.0)
        self.potential += effective_input
        
        # 3. Génération des Spikes avec Seuil Adaptatif
        self.spikes = self.potential >= self.threshold
        
        # 4. MISE À JOUR HOMÉOSTASIE (Le seuil monte si ça spike trop)
        self.threshold[self.spikes] += 0.05
        self.threshold = (self.threshold * 0.99) + (self.base_threshold * 0.01)
        
        # 5. MISE À JOUR FATIGUE
        self.fatigue[self.spikes] += 0.4
        self.fatigue *= 0.85 # Récupération lente
        
        # 6. Trace de sortie et Reset
        self.trace_post = (self.trace_post * self.trace_decay) + self.spikes.astype(np.float32)
        self.potential[self.spikes] = 0
        
        self.timestamp += 1
        return self.spikes.astype(np.float32)

    def apply_stdp(self, learning_rate=0.001, mod_sens=1.0):
        # Règle STDP classique
        ltp = np.outer(self.spikes, self.trace_pre)
        ltd = np.outer(self.trace_post, self.trace_pre) * 0.5
        self.weights += (ltp - ltd) * learning_rate * mod_sens
        self.weights = np.clip(self.weights, -1.0, 1.0)

class ExecutiveCortex(BioTensor):
    def __init__(self, shape, actions, **kwargs):
        super().__init__(shape, **kwargs)
        self.actions = actions
        self.pool_size = shape[0] // len(actions)
        self.dopamine = 1.0

    def get_decision(self):
        """ Vote par groupe de neurones """
        votes = []
        for i in range(len(self.actions)):
            start, end = i * self.pool_size, (i + 1) * self.pool_size
            votes.append(np.mean(self.potential[start:end]))
        return self.actions[np.argmax(votes)]

    def apply_reinforcement(self, reward):
        self.dopamine = np.clip(self.dopamine + reward, 0.1, 3.0)
        self.apply_stdp(learning_rate=0.005, mod_sens=self.dopamine)

class BrocaArea(BioTensor):
    def __init__(self, shape, **kwargs):
        super().__init__(shape, **kwargs)
        self.cooldown = 0
        self.last_word = None

    def process_speech(self, vocabulary):
        """ Séquenceur de mots avec cooldown pour éviter le bégaiement """
        if self.cooldown > 0:
            self.cooldown -= 1
            return None

        # On cherche le mot le plus proche de la trace actuelle de Broca
        current_sig = self.trace_pre
        best_word = None
        max_score = 0
        
        for name, pattern in vocabulary.items():
            score = np.dot(current_sig, pattern[:len(current_sig)])
            if score > max_score:
                max_score = score
                best_word = name
        
        # Si le score est bon et que c'est un mot différent du dernier dit
        if max_score > 3.0 and best_word != self.last_word:
            self.last_word = best_word
            self.cooldown = 20 # Pause entre les mots
            return best_word
        
        if max_score < 1.0: self.last_word = None
        return None

class BioBrain:
    def __init__(self):
        # Couches
        self.sensory = BioTensor((400, 100), name="Sensory")
        self.associative = BioTensor((400, 400), name="Associative", threshold=0.8)
        self.executive = ExecutiveCortex((100, 400), actions=["STABLE", "GAUCHE", "DROITE", "ALERTE"])
        self.limbic = BioTensor((100, 400), name="Limbic", threshold=1.2)
        self.broca = BrocaArea((100, 400), name="Broca")
        
        # État interne
        self.arousal = 0.5
        self.internal_feedback = np.zeros(400, dtype=np.float32)
        self.speech_buffer = []

    def forward(self, raw_input):
        """ 
        Propagation directe utilisée par le compilateur pour l'entraînement.
        """
        s_out = self.sensory.forward(raw_input)
        a_out = self.associative.forward(s_out)
        e_out = self.executive.forward(a_out)
        # On fait aussi passer dans Broca et le Limbique pour qu'ils voient le signal
        self.broca.forward(a_out)
        self.limbic.forward(a_out)
        
        return s_out, a_out, e_out

    def process_internal_thought(self, external_input=None, vocabulary={}):
        # 1. Gestion de la vigilance et de l'entrée
        is_external = external_input is not None
        if not is_external:
            # Rêverie (bruit de fond)
            x_in = (np.random.rand(100) > (1.0 - 0.01 * self.arousal)).astype(np.float32)
            f_weight = 0.2
        else:
            x_in = external_input
            f_weight = 0.05 # On baisse le feedback pour écouter l'extérieur
            self.arousal = min(1.0, self.arousal + 0.1)

        # 2. Propagation Sensory -> Associative (avec Feedback)
        s_out = self.sensory.forward(x_in)
        a_in = np.clip(s_out + (self.internal_feedback * f_weight), 0, 1)
        a_out = self.associative.forward(a_in)
        self.internal_feedback = a_out
        
        # 3. Contrôle Limbique (Lois)
        l_out = self.limbic.forward(a_out)
        if np.mean(l_out) > 0.15:
            self.executive.potential *= 0.0 # Blocage total
            self.executive.apply_reinforcement(-0.3)

        # 4. Exécution et Parole
        self.executive.forward(a_out)
        self.broca.forward(a_out)
        
        new_word = self.broca.process_speech(vocabulary)
        if new_word:
            self.speech_buffer.append(new_word)
            if len(self.speech_buffer) > 5: self.speech_buffer.pop(0)

        # 5. Homéostasie de la vigilance
        act = np.mean(a_out)
        if act > 0.12: self.arousal *= 0.98
        elif act < 0.02: self.arousal *= 1.02

        return self.executive.get_decision()

    def get_current_thought(self, vocabulary):
        """ Décodeur de debug pour l'interface """
        sig = self.sensory.trace_pre
        if not np.any(sig > 0.1): return "Repos..."
        
        best, max_s = "Inconnu", 0
        for name, pattern in vocabulary.items():
            score = np.dot(sig, pattern)
            if score > max_s:
                max_s = score
                best = name
        return best if max_s > 2.5 else "Rêverie..."
    
# Dans brain_architecture.py

class VisualPrimary(BioTensor):
    def __init__(self, shape=(200, 100), **kwargs):
        super().__init__(shape, name="Visual_V1", **kwargs)
        # On peut pré-câbler des détecteurs de bords (facultatif pour le moment)

class AuditoryPrimary(BioTensor):
    def __init__(self, shape=(200, 100), **kwargs):
        super().__init__(shape, name="Auditory_A1", **kwargs)
        # Sensible aux séquences temporelles