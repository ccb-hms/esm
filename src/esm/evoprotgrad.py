"""
EvoProtGrad: protein engineering through directed evolution
https://huggingface.co/blog/AmelieSchreiber/directed-evolution-with-esm2
Andreas Werdich
2024 Center for Computational Biomedicine
Harvard Medical School
"""
import os
import logging
import pandas as pd
import tempfile

import evo_prot_grad
import torch
from transformers import AutoTokenizer, EsmForMaskedLM

logger = logging.getLogger(name=__name__)

torch.set_float32_matmul_precision(precision='high')

# https://huggingface.co/facebook/esm2_t33_650M_UR50D
esm_checkpoints = {
    't48_15B': 'facebook/esm2_t48_15B_UR50D',
    't36_3B': 'facebook/esm2_t36_3B_UR50D',
    't33_650M': 'facebook/esm2_t33_650M_UR50D',
    't30_150M': 'facebook/esm2_t30_150M_UR50D',
    't12_35M': 'facebook/esm2_t12_35M_UR50D',
    't6/8M': 'facebook/esm2_t6_8M_UR50D',
    'default': 'facebook/esm2_t30_150M_UR50D'
}


def set_expert(name='esm', checkpoint='default', device=None, cache_dir=None):
    checkpoint = esm_checkpoints.get(checkpoint)
    expert = evo_prot_grad.get_expert(
        expert_name=name,
        model=EsmForMaskedLM.from_pretrained(checkpoint, cache_dir=cache_dir),
        tokenizer=AutoTokenizer.from_pretrained(checkpoint, cache_dir=cache_dir),
        temperature=0.95,
        device=device)
    return expert


def torch_device():
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device_id = torch.cuda.current_device()
        device_dict = {
            'device_id': device_id,
            'device': torch.device(f'cuda:{device_id}'),
            'device_name': torch.cuda.get_device_name(device_id),
            'cudnn_version': torch.backends.cudnn.version()
        }
    else:
        device_dict = {
            'device_id': None,
            'device': torch.device('cpu'),
            'device_name': 'cpu',
            'cudnn_version': None
        }
    device_dict.update({'torch_version': torch.__version__})
    return device_dict


class EvoProtGrad:
    def __init__(self, name='esm', device=None, expert=None, cache_dir=None):
        self.name = name
        if device is None:
            device = torch_device().get('device')
        self.device = device
        if expert is None:
            expert = set_expert(name=name, 
                                checkpoint='default', 
                                device=self.device, 
                                cache_dir=cache_dir)
        self.expert = expert
    
    def single_evolute(self, raw_protein_sequence, **kwargs):
        variants, scores = self.evolute(raw_protein_sequence=raw_protein_sequence)
        var_df_list = []
        for evolution, var_protein_sequence in enumerate(variants):
            var_protein_sequence = var_protein_sequence[0].replace(' ', '')
            dif_pos_list = [i for i in range(len(raw_protein_sequence)) if
                            var_protein_sequence[i] != raw_protein_sequence[i]]
            dif_raw_list = [raw_protein_sequence[i] for i in dif_pos_list]
            dif_var_list = [var_protein_sequence[i] for i in dif_pos_list]
            var_dict = {'variant': [evolution],
                        'score': scores[evolution],
                        'pos': [dif_pos_list],
                        'source': [dif_raw_list],
                        'target': [dif_var_list],
                        'sequence': [var_protein_sequence]}
            var_df_list.append(pd.DataFrame(var_dict, index=[evolution]))
        var_df = pd.concat(var_df_list, axis=0). \
            sort_values(by='score', ascending=False). \
            reset_index(drop=True)
        return var_df

    def evolute(self, raw_protein_sequence):
        fasta_format_sequence = f'>Input_Sequence\n{raw_protein_sequence}'
        with tempfile.NamedTemporaryFile(mode='w+t', delete=False) as temp_fasta_path:
            temp_fasta_path.write(fasta_format_sequence)
            temp_fasta_path.close()
            directed_evolution = evo_prot_grad.DirectedEvolution(
                wt_fasta=temp_fasta_path.name,  # path to the temporary FASTA file
                output='all',  # can be 'best', 'last', or 'all' variants
                experts=[self.expert],  # list of experts, in this case only ESM-2
                parallel_chains=1,  # number of parallel chains to run
                n_steps=20,  # number of MCMC steps per chain
                max_mutations=10,  # maximum number of mutations per variant
                verbose=False  # print debug info
            )
            # Run the evolution process
            variants, scores = directed_evolution()
        return variants, scores
