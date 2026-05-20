import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
import pickle
import os
import traceback
from brain_architecture import BioBrain # Assure-toi que le nom du fichier est correct

class MultiMonitorV4:
    def __init__(self):
        self.brain = BioBrain()
        self.vocabulary = {} # Chargé depuis le fichier .bio
        self.is_running = False
        
        self.root = tk.Tk()
        self.root.title("Bio-Intelligence Monitor - Chargeur de Conscience")
        self.root.geometry("1100x850")
        self.root.configure(bg="#0a0a0a")

        self.setup_ui()

    def setup_ui(self):
        # --- Barre Latérale ---
        sidebar = tk.Frame(self.root, bg="#1a1a1a", width=250)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(sidebar, text="SYSTÈME", fg="#00FF41", bg="#1a1a1a", font=("Courier", 12, "bold")).pack(pady=10)
        
        # Bouton pour charger le cerveau compilé
        tk.Button(sidebar, text="CHARGER CERVEAU (.bio)", command=self.load_compiled_brain, 
                  bg="#004400", fg="white").pack(fill=tk.X, padx=10, pady=5)
        
        self.lbl_brain_status = tk.Label(sidebar, text="Cerveau: VIERGE", fg="orange", bg="#1a1a1a")
        self.lbl_brain_status.pack()

        ttk.Separator(sidebar).pack(fill=tk.X, pady=15)

        # Boutons d'injection de concepts (Se remplira après chargement)
        tk.Label(sidebar, text="INJECTION CONCEPTS", fg="white", bg="#1a1a1a").pack()
        self.concept_frame = tk.Frame(sidebar, bg="#1a1a1a")
        self.concept_frame.pack(fill=tk.BOTH, expand=True, padx=5)

        self.btn_run = tk.Button(sidebar, text="LANCER PENSÉE", command=self.toggle, bg="#00FF41", state=tk.DISABLED)
        self.btn_run.pack(fill=tk.X, padx=10, pady=10)

        # --- Canvas de Visualisation ---
        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0)
        self.canvas.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        self.lbl_decision = tk.Label(sidebar, text="DÉCISION: ...", fg="#00FF41", bg="#1a1a1a", font=("Arial", 10))
        self.lbl_decision.pack(pady=10)

        # AJOUTE CECI :
        self.lbl_thought = tk.Label(sidebar, text="PENSÉE: ...", fg="yellow", bg="#1a1a1a", font=("Courier", 10, "bold"))
        self.lbl_thought.pack(pady=10)
        
        # Configuration des grilles
        self.layers_ui = {
            "executive":   {"offset": 50,  "size": 250, "dim": 10, "rects": []}, # 10x10=100
            "associative": {"offset": 320, "size": 250, "dim": 20, "rects": []}, # 20x20=400
            "sensory":     {"offset": 590, "size": 250, "dim": 20, "rects": []}  # 20x20=400
        }
        self.create_grids()

    def create_grids(self):
        for name, info in self.layers_ui.items():
            tk.Label(self.canvas, text=name.upper(), fg="white", bg="black").place(x=10, y=info["offset"]-20)
            cell_size = info["size"] // info["dim"]
            for i in range(info["dim"]):
                for j in range(info["dim"]):
                    x = 100 + j * cell_size
                    y = info["offset"] + i * cell_size
                    r = self.canvas.create_rectangle(x, y, x+cell_size-1, y+cell_size-1, fill="#050505", outline="#111")
                    info["rects"].append(r)

    def load_compiled_brain(self):
        path = filedialog.askopenfilename(filetypes=[("BioBrain Files", "*.bio")])
        if not path: return
        
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                
            # INJECTION DES POIDS DANS L'ARCHITECTURE
            self.brain.sensory.weights = data['weights_s']
            self.brain.associative.weights = data['weights_a']
            self.brain.executive.weights = data['weights_e']
            self.brain.limbic.weights = data['weights_l']
            if 'weights_b' in data: # Si Broca a été compilé
                self.brain.broca.weights = data['weights_b']
            
            self.vocabulary = data['vocabulary']
            
            # Mise à jour UI
            self.lbl_brain_status.config(text=f"Cerveau: {os.path.basename(path)}", fg="#00FF41")
            self.btn_run.config(state=tk.NORMAL)
            self.update_concept_buttons()
            print("Cerveau compilé injecté avec succès.")

        except Exception as e:
            print(traceback.format_exc())
            self.lbl_brain_status.config(text="ERREUR CHARGEMENT", fg="red")

    def update_concept_buttons(self):
        """ Crée des boutons pour chaque concept appris par le cerveau """
        for widget in self.concept_frame.winfo_children():
            widget.destroy()
        
        for concept_name in self.vocabulary.keys():
            btn = tk.Button(self.concept_frame, text=concept_name, 
                           command=lambda c=concept_name: self.inject_concept(c),
                           bg="#333", fg="white", font=("Arial", 8))
            btn.pack(fill=tk.X, pady=1)

    def inject_concept(self, name):
        """ Injecte le concept de manière soutenue (pendant 10 cycles) """
        if name in self.vocabulary:
            pattern = self.vocabulary[name]
            # On crée une petite boucle d'injection forcée
            def burst(count):
                if count > 0:
                    self.brain.process_internal_thought(external_input=pattern)
                    self.root.after(20, lambda: burst(count-1))
            
            burst(10) # Injecte le motif pendant 10 cycles consécutifs
            print(f"Injection intensive de : {name}")
            
    def toggle(self):
        self.is_running = not self.is_running
        self.btn_run.config(text="STOP" if self.is_running else "LANCER PENSÉE")
        if self.is_running: self.loop()

    def loop(self):
        if not self.is_running: return
        
        # Mode Pensée Spontanée : on passe None ou un bruit rose très léger
        # Cela utilise la méthode process_internal_thought que nous avons définie
        decision = self.brain.process_internal_thought(external_input=None)
        thought = self.brain.get_current_thought(self.vocabulary)
        self.lbl_thought.config(text=f"PENSÉE: {thought}")
        # Mise à jour visuelle (on récupère les spikes des couches du cerveau)
        self.update_layer_viz("sensory", self.brain.sensory.spikes, self.brain.sensory.potential)
        self.update_layer_viz("associative", self.brain.associative.spikes, self.brain.associative.potential)
        self.update_layer_viz("executive", self.brain.executive.spikes, self.brain.executive.potential)
        
        self.root.after(50, self.loop)

    def update_layer_viz(self, layer_name, spikes, potentials):
        rects = self.layers_ui[layer_name]["rects"]
        for i, r in enumerate(rects):
            if i < len(spikes) and spikes[i]:
                col = "#00FF41"
            else:
                v = int(max(0, min(potentials[i] * 100, 255)))
                col = f"#{v:02x}0000"
            self.canvas.itemconfig(r, fill=col)

if __name__ == "__main__":
    app = MultiMonitorV4()
    app.root.mainloop()