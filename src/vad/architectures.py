import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class TrajectoryEmbedding(nn.Module):
    def __init__(self, n_lat_bins, n_lon_bins, n_sog_bins, n_cog_bins, embed_dim):
        super(TrajectoryEmbedding, self).__init__()

        # Create embedding layers for each feature
        self.lat_embed = nn.Embedding(n_lat_bins, embed_dim)
        self.lon_embed = nn.Embedding(n_lon_bins, embed_dim)
        self.sog_embed = nn.Embedding(n_sog_bins, embed_dim)
        self.cog_embed = nn.Embedding(n_cog_bins, embed_dim)

    def forward(self, x):
        # Input is a dictionary where each value is a feature with shape: (batch_size, seq_len)
        # The values are the index value of where the bin is 'hot'

        # Assuming input is a dict for each feature
        lat_emb = self.lat_embed(x['lat'])  # Outputs [batch_size, seq_len, embed_dim]
        lon_emb = self.lon_embed(x['lon'])
        sog_emb = self.sog_embed(x['sog'])
        cog_emb = self.cog_embed(x['cog'])

        # Concatenate the embeddings
        combined = torch.cat([lat_emb, lon_emb, sog_emb, cog_emb], dim=-1)
        return combined

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len, dropout):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix optimized for batch_first=True
        pe = torch.zeros(1, max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        # Register as a buffer (not a parameter)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add fixed positional encodings to sequence embeddings.

        Args:
            x: Tensor of shape [batch_size, seq_length, embedding_dim]
        Returns:
            Tensor of same shape with positional encodings added
        """
        # Add the fixed positional encoding
        return x + self.pe

class TrajectoryTransformerEncoder(nn.Module):
    def __init__(self,
                 embed_dim,
                 n_head,
                 n_layers,
                 dropout,
                 max_seq_len):
        super(TrajectoryTransformerEncoder, self).__init__()

        # Projection to adjust dimension for transformer
        self.d_model = embed_dim * 4

        # Positional encoding for transformer
        self.pos_encoder = PositionalEncoding(self.d_model, max_seq_len, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,  # Concatenated embeddings
            nhead=n_head,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)

    def forward(self, x):
        """
        :param x: Concatenated embeddings with shape (batch_size, sequence_length, 4*embed_dim)
        :return: The output logits with shape (batch_size, seq_length, d_model)
        """

        # Set up mask
        seq_len = x.shape[1]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device) * float('-inf'),
            diagonal=1
        )

        # Pass through transformer
        h = self.transformer(x, mask=causal_mask) # Logits

        return h

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, dropout):
        super(Autoencoder, self).__init__()

        hidden_dim1 = max(input_dim // 4, latent_dim * 8)  # First compression
        hidden_dim2 = max(input_dim // 16, latent_dim * 2)  # Second compression

        # Ensure dimensions are reasonable
        hidden_dim1 = min(hidden_dim1, input_dim // 2)
        hidden_dim2 = max(hidden_dim2, latent_dim * 2)


        # Encoder
        self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim1),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim1, hidden_dim2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim2, latent_dim),
                    nn.ReLU()
                )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class GMMEstimationNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_components, eps):
        super(GMMEstimationNetwork, self).__init__()
        self.num_components = num_components
        self.input_dim = input_dim
        self.eps = eps

        # MLP to predict component membership probabilities
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2), # bottleneck
            nn.ReLU(),
            nn.Linear(hidden_dim//2, num_components),
        )

    def forward(self, z_batch):
        batch_size = z_batch.size(0)

        # Predict the membership probabilities for each component
        logits = self.mlp(z_batch)
        gamma = F.softmax(logits, dim=1)  # Shape: [batch_size, num_components]

        # Calculate GMM parameters based on estimated memberships
        # Mixture weights with small epsilon to prevent zero weights
        phi = gamma.mean(dim=0) + self.eps / self.num_components
        phi = phi / phi.sum()  # Renormalize

        # Component means - vectorized
        gamma_expanded = gamma.t().unsqueeze(2)
        z_batch_expanded = z_batch.unsqueeze(0)

        # Weighted sum for all components at once
        weighted_sum = torch.sum(gamma_expanded * z_batch_expanded, dim=1)
        gamma_sum = gamma.sum(dim=0).unsqueeze(1) + self.eps
        mu = weighted_sum / gamma_sum

        # Component covariances - with improved numerical stability
        sigma = torch.zeros(self.num_components, self.input_dim, self.input_dim,
                           device=z_batch.device)

        for c in range(self.num_components):
            # Center the data
            z_centered = z_batch - mu[c].unsqueeze(0)

            # Compute weighted covariance
            gamma_c = gamma[:, c].unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1]

            # Outer products: [batch_size, input_dim, input_dim]
            outer_products = torch.bmm(
                z_centered.unsqueeze(2),
                z_centered.unsqueeze(1)
            )

            # Weight by gamma
            weighted_outer = gamma_c * outer_products

            # Sum over batch dimension and normalize
            sigma[c] = weighted_outer.sum(dim=0) / (gamma[:, c].sum() + self.eps)

            # Add stronger regularization to diagonal for numerical stability
            # This ensures the covariance matrix is positive definite
            sigma[c] = sigma[c] + torch.eye(self.input_dim, device=z_batch.device) * self.eps

        return gamma, phi, mu, sigma

    def compute_energy(self, z, phi, mu, sigma):
        """
        Vectorized computation of energy (negative log-likelihood) for each sample
        """
        batch_size = z.size(0)
        num_components = self.num_components
        input_dim = self.input_dim

        # Prepare for vectorized computation
        z_expanded = z.unsqueeze(1).expand(batch_size, num_components, input_dim)
        z_centered = z_expanded - mu.unsqueeze(0)

        # Ensure numerical stability of covariance matrices
        sigma_regularized = sigma + torch.eye(input_dim, device=z.device).unsqueeze(0) + self.eps

        # Compute log determinant for each component
        sign, logdet = torch.slogdet(2 * np.pi * sigma_regularized)

        log_det_term = 0.5 * logdet

        # Compute inverse using Cholesky decomposition for better stability
        try:
            # Cholesky decomposition
            L = torch.linalg.cholesky(sigma_regularized)

            # Solve L @ L^T @ x = z_centered for x
            # First solve L @ y = z_centered for y
            z_centered_flat = z_centered.reshape(batch_size * num_components, input_dim, 1)
            L_expanded = L.unsqueeze(0).expand(batch_size, -1, -1, -1).reshape(
                batch_size * num_components, input_dim, input_dim
            )

            # Use triangular solve for efficiency
            y = torch.linalg.solve_triangular(L_expanded, z_centered_flat, upper=False)

            # Mahalanobis distance is ||y||^2
            mahalanobis_dist = torch.sum(y.squeeze(-1) ** 2, dim=1).reshape(batch_size, num_components)

        except:
            # Fallback to direct inverse if Cholesky fails
            inv_sigma = torch.inverse(sigma_regularized)
            z_centered_flat = z_centered.reshape(batch_size * num_components, input_dim)
            inv_sigma_expanded = inv_sigma.unsqueeze(0).expand(
                batch_size, num_components, input_dim, input_dim
            ).reshape(batch_size * num_components, input_dim, input_dim)

            mahalanobis_part1 = torch.bmm(
                z_centered_flat.unsqueeze(1),
                inv_sigma_expanded
            ).squeeze(1)

            mahalanobis_dist = torch.sum(
                mahalanobis_part1 * z_centered_flat,
                dim=1
            ).reshape(batch_size, num_components)

        # Compute the log of component probabilities
        log_probs = (
            torch.log(phi + self.eps).unsqueeze(0) -
            0.5 * mahalanobis_dist -
            log_det_term.unsqueeze(0)
        )

        # Use the log-sum-exp trick for numerical stability
        max_log_probs = torch.max(log_probs, dim=1, keepdim=True)[0]
        exp_sum = torch.sum(
            torch.exp(log_probs - max_log_probs),
            dim=1
        )
        log_sum = max_log_probs.squeeze(1) + torch.log(exp_sum + self.eps)

        # Energy is negative log-likelihood
        energies = -log_sum

        return energies

class STAD(nn.Module):
    def __init__(self,
                 n_lat_bins,
                 n_lon_bins,
                 n_sog_bins,
                 n_cog_bins,
                 max_seq_len,
                 embed_dim,
                 dropout,
                 nhead_te,
                 n_layers_te,
                 latent_dim_ae,
                 hidden_dim_gmm,
                 n_components_gmm,
                 eps_gmm,
                 n_weather_vars,
                 size_window=10):

        super(STAD, self).__init__()

        # Embedding module
        self.embedding = TrajectoryEmbedding(n_lat_bins=n_lat_bins,
                                             n_lon_bins=n_lon_bins,
                                             n_sog_bins=n_sog_bins,
                                             n_cog_bins=n_cog_bins,
                                             embed_dim=embed_dim)

        # Transformer Encoder
        self.transenc = TrajectoryTransformerEncoder(embed_dim=embed_dim,
                                                     n_head=nhead_te,
                                                     n_layers=n_layers_te,
                                                     dropout=dropout,
                                                     max_seq_len=max_seq_len)

        # Layer normalization for the outputs of the Transformer Encoder
        self.layer_norm = nn.LayerNorm(self.transenc.d_model)

        # Linear output projection layers for the Transformer Encoder
        self.lat_out = nn.Linear(self.transenc.d_model, n_lat_bins)
        self.lon_out = nn.Linear(self.transenc.d_model, n_lon_bins)
        self.sog_out = nn.Linear(self.transenc.d_model, n_sog_bins)
        self.cog_out = nn.Linear(self.transenc.d_model, n_cog_bins)

        # Autoencoder
        self.ae = Autoencoder(input_dim=self.transenc.d_model * size_window, # Flatten window
                              latent_dim=latent_dim_ae,
                              dropout=dropout)

        # Gaussian Mixture Model implementation via MLP
        self.n_weather_vars = n_weather_vars
        self.gmm = GMMEstimationNetwork(input_dim=latent_dim_ae + 2 + self.n_weather_vars * 2,
                                        hidden_dim=hidden_dim_gmm,
                                        num_components=n_components_gmm,
                                        eps=eps_gmm)

    def forward(self, src, tgt, weather_stats=None, training=True):
        # Embed {t_1 - T-1}
        x = self.embedding(src)
        h = self.transenc(x)

        # Pass through projection layer
        h_norm = self.layer_norm(h)
        lat_logits = self.lat_out(h_norm)
        lon_logits = self.lon_out(h_norm)
        sog_logits = self.sog_out(h_norm)
        cog_logits = self.cog_out(h_norm)

        criterion = nn.CrossEntropyLoss(reduction='none')
        if training:
            # Loss for Transformer Encoder in training phase
            lat_loss_tr = criterion(lat_logits.transpose(1, 2), tgt['lat'])
            lon_loss_tr = criterion(lon_logits.transpose(1, 2), tgt['lon'])
            sog_loss_tr = criterion(sog_logits.transpose(1, 2), tgt['sog'])
            cog_loss_tr = criterion(cog_logits.transpose(1, 2), tgt['cog'])
            ce_per_timestep = lat_loss_tr + lon_loss_tr + sog_loss_tr + cog_loss_tr  # [batch, seq_len]
            l = ce_per_timestep.sum(dim=1)  # [batch]
        else:
            # Loss for Transformer Encoder in testing phase
            lat_loss_ts = criterion(lat_logits.transpose(1, 2), tgt['lat']).max()
            lon_loss_ts = criterion(lon_logits.transpose(1, 2), tgt['lon']).max()
            sog_loss_ts = criterion(sog_logits.transpose(1, 2), tgt['sog']).max()
            cog_loss_ts = criterion(cog_logits.transpose(1, 2), tgt['cog']).max()
            l = lat_loss_ts + lon_loss_ts + sog_loss_ts + cog_loss_ts
            return l

        # Save original shape for reconstruction
        batch_size, seq_len, feat_dim = h.shape
        h_flat = h.reshape(batch_size, -1)  # Flatten to (batch_size, seq_len*feat_dim)

        # Pass Transformer Encoder Latent representation through Auto Encoder
        h_e, h_hat = self.ae(h_flat)

        # Reconstruction error for Auto Encoder
        reconstruction_criterion = nn.MSELoss(reduction='none')
        d_h = reconstruction_criterion(h_flat, h_hat).mean(dim=1)

        # Construct z
        z = torch.cat([h_e, l.unsqueeze(-1), d_h.unsqueeze(-1), weather_stats], dim=-1)

        # Pass through GMM
        gamma, phi, mu, sigma = self.gmm(z)
        energy = self.gmm.compute_energy(z, phi, mu, sigma)

        return l, energy, d_h, sigma
