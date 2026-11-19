from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class PCAReducer:
    """Zero-leakage feature standardization and orthogonal PCA projection."""

    def __init__(self, n_components: int = 150, use_pca: bool = True):
        self.n_components = n_components
        self.use_pca = use_pca
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, svd_solver="auto", random_state=42) if use_pca else None

    def fit_transform(self, X_train):
        X_scaled = self.scaler.fit_transform(X_train)
        if not self.use_pca:
            return X_scaled, 1.0
        X_pca = self.pca.fit_transform(X_scaled)
        explained_var = float(self.pca.explained_variance_ratio_.sum())
        return X_pca, explained_var

    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        if not self.use_pca:
            return X_scaled
        return self.pca.transform(X_scaled)
