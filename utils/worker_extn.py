import gc
import time
import torch

def _stateless_init_process_group(master_address, master_port, rank, world_size, device):
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup
    pg = StatelessProcessGroup.create(
        host=master_address, port=master_port, rank=rank, world_size=world_size
    )
    return PyNcclCommunicator(pg, device=device)

class WorkerExtension:
    @torch.no_grad()
    def perturb_self_weights(self, seed, noise_scale, negate=False):
        scale = float(noise_scale)
        sign = -1.0 if negate else 1.0

        for _, p in self.model_runner.model.named_parameters():
            gen = torch.Generator(device=p.device)
            gen.manual_seed(int(seed))
            noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
            p.data.add_(sign * scale * noise)
            del noise

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return True

    @torch.no_grad()
    def restore_self_weights(self, seed, SIGMA):
        for _, p in self.model_runner.model.named_parameters():
            gen = torch.Generator(device=p.device)
            gen.manual_seed(int(seed))
            noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
            p.data.add_(-float(SIGMA) * noise)
            del noise

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return True

    @torch.no_grad()
    def compute_weighted_noise_frob_norm(self, seed_coeffs):
        total_sq = 0.0

        for _, p in self.model_runner.model.named_parameters():
            acc = torch.zeros_like(p.data, dtype=torch.float32, device=p.device)

            for seed, coeff in seed_coeffs:
                gen = torch.Generator(device=p.device)
                gen.manual_seed(int(seed))
                noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
                acc.add_(float(coeff) * noise.float())
                del noise

            total_sq += float(torch.sum(acc * acc).item())
            del acc

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()

        return total_sq ** 0.5

    def init_inter_engine_group(self, master_address: str, master_port: int, rank: int, world_size: int):
        self.inter_pg = _stateless_init_process_group(
            master_address, master_port, rank, world_size, self.device
        )
        return True

    @torch.no_grad()
    def broadcast_all_weights(self, src_rank: int):
        for _, p in self.model_runner.model.named_parameters():
            self.inter_pg.broadcast(p, src=int(src_rank), stream=torch.cuda.current_stream())

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

    def save_self_weights_to_disk(self, filepath):
        state_dict_to_save = {}
        for name, p in self.model_runner.model.named_parameters():
            state_dict_to_save[name] = p.detach().cpu()

        torch.save(state_dict_to_save, filepath)
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        time.sleep(0.1)
        return True

    def load_weights_from_disk(self, filepath):
        state_dict = torch.load(filepath, map_location=self.device)
        for name, p in self.model_runner.model.named_parameters():
            p.data.copy_(state_dict[name].to(self.device))

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        time.sleep(0.1)
        return True
