class EarlyStopping:
    """
    Early stopping basé sur une métrique de validation.

    Args:
        patience (int): Nombre d'époques sans amélioration.
        min_delta (float): Amélioration minimale pour être considérée.
        mode (str): 'min' pour une loss, 'max' pour une métrique (accuracy, f1).
    """

    def __init__(self, patience=3, min_delta=0.0, mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        improved = (
            score < self.best_score - self.min_delta
            if self.mode == "min"
            else score > self.best_score + self.min_delta
        )

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop
