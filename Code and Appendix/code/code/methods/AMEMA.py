import torch
import torch.nn as nn
import torch.nn.functional as F
from methods.template import LDLTemplate, kaiming_normal_init_net

class AMEMA(LDLTemplate):
    def __init__(
        self,
        num_feature,
        num_classes,
        loss_func,
        hidden_dim=100,
        lambda1=0.1,
        lambda2=0.1,
        lambda3=0.1,
        lr=1e-3,
        weight_decay=1e-4,
        adjust_lr=False,
        gradient_clip_value=4.0,
        max_epoch=300,
        verbose=False,
        device="cuda:0",
    ):
        super().__init__(
            num_feature,
            num_classes,
            adjust_lr=adjust_lr,
            gradient_clip_value=gradient_clip_value,
            max_epoch=max_epoch,
            verbose=verbose,
            device=device,
        )

        self.loss_func = loss_func
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        self.encoder_x = nn.Sequential(
            nn.Linear(num_feature, hidden_dim),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
        )
        self.encoder_x_mu  = nn.Linear(hidden_dim, hidden_dim)
        self.encoder_x_var = nn.Linear(hidden_dim, hidden_dim)

        self.encoder_ys = nn.Sequential(
            nn.Linear(num_classes + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.encoder_ys_mu  = nn.Linear(hidden_dim, hidden_dim)
        self.encoder_ys_var = nn.Linear(hidden_dim, hidden_dim)

        self.encoder_yns = nn.Sequential(
            nn.Linear(num_classes + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.encoder_yns_mu  = nn.Linear(hidden_dim, hidden_dim)
        self.encoder_yns_var = nn.Linear(hidden_dim, hidden_dim)

        self.decoder_pred_sig = nn.Linear(hidden_dim, num_classes)
        self.decoder_pred_ns  = nn.Linear(hidden_dim, num_classes)

        self.register_buffer("loss_s_ema", torch.tensor(1.0))
        self.register_buffer("loss_ns_ema", torch.tensor(1.0))

        self.branch_sig_params = list(self.encoder_ys.parameters()) + \
                                  list(self.encoder_ys_mu.parameters()) + \
                                  list(self.encoder_ys_var.parameters())
        self.branch_ns_params  = list(self.encoder_yns.parameters()) + \
                                  list(self.encoder_yns_mu.parameters()) + \
                                  list(self.encoder_yns_var.parameters())
        self.pred_sig_params   = list(self.decoder_pred_sig.parameters())
        self.pred_ns_params    = list(self.decoder_pred_ns.parameters())

        all_branch = set(self.branch_sig_params + self.branch_ns_params +
                         self.pred_sig_params + self.pred_ns_params)
        self.shared_params = [p for p in self.parameters() if p not in all_branch]

        self.m_sig = 0.90
        self.m_ns = 0.95

        self.optimizer = torch.optim.SGD(
            [
                {'params': self.shared_params,     'momentum': 0.85},
                {'params': self.branch_sig_params, 'momentum': self.m_sig},
                {'params': self.branch_ns_params,  'momentum': self.m_ns},
                {'params': self.pred_sig_params,   'momentum': self.m_sig},
                {'params': self.pred_ns_params,    'momentum': self.m_ns},
            ],
            lr=lr,
            weight_decay=weight_decay,
        )

        self.to(self.device)

    @staticmethod
    def _get_branch_targets(y, num_classes, eps=1e-6):
        y_sum = y.sum(dim=1, keepdim=True).clamp(min=eps)
        y_norm = y / y_sum
        idx = y_norm.argmax(dim=1)
        y_sig = F.one_hot(idx, num_classes).float().to(y.device)
        y_res = (y_norm - y_sig).clamp(min=0.0)
        y_ns = y_res / y_res.sum(dim=1, keepdim=True).clamp(min=eps)
        y_ns[y_res.sum(dim=1) == 0] = 0
        return y_sig, y_ns, y_norm

    def set_forward_loss(self, x, y):
        eps = 1e-6
        alpha = 4.0
        base = 0.1

        xx = self.encoder_x(x)
        mu_x = self.encoder_x_mu(xx)
        var_x = F.softplus(self.encoder_x_var(xx)) + eps

        y_sig, y_ns, y_norm = self._get_branch_targets(y, self.num_classes, eps)

        inp_s = torch.cat([y_sig, mu_x.detach()], dim=1)
        ys = self.encoder_ys(inp_s)
        mu_s = self.encoder_ys_mu(ys)
        var_s = F.softplus(self.encoder_ys_var(ys)) + eps

        inp_ns = torch.cat([y_ns, mu_x.detach()], dim=1)
        yns = self.encoder_yns(inp_ns)
        mu_ns = self.encoder_yns_mu(yns)
        var_ns = F.softplus(self.encoder_yns_var(yns)) + eps

        kl_s = 0.5 * torch.sum(
            torch.log(var_x / var_s) + var_s / var_x + (mu_x - mu_s).pow(2) / var_x - 1,
            dim=1
        )
        kl_ns = 0.5 * torch.sum(
            torch.log(var_x / var_ns) + var_ns / var_x + (mu_x - mu_ns).pow(2) / var_x - 1,
            dim=1
        )
        loss_s = kl_s.mean()
        loss_ns = kl_ns.mean()

        with torch.no_grad():
            self.loss_s_ema.mul_(self.m_sig).add_((1 - self.m_sig) * loss_s)
            self.loss_ns_ema.mul_(self.m_ns).add_((1 - self.m_ns) * loss_ns)
        r = torch.clamp(self.loss_ns_ema / (self.loss_s_ema + eps), 0.1, 10.0)
        w_ns = base + (1 - base) * torch.sigmoid(alpha * (r - 1.0))
        w_s = 1.0 - w_ns

        loss = w_s * loss_s + w_ns * loss_ns

        components = {
            'loss_s': loss_s.item(),
            'loss_ns': loss_ns.item(),
            'w_ns': w_ns.item(),
            'momentum_sig': self.m_sig,
            'momentum_ns': self.m_ns,
            'ema_s': self.loss_s_ema.item(),
            'ema_ns': self.loss_ns_ema.item(),
        }
        return loss, components

    def forward(self, x):
        h = self.encoder_x(x)
        logits = self.decoder_pred_sig(F.relu(h))
        return F.softmax(logits, dim=1)
