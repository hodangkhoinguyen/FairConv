from __future__ import annotations
import warnings
warnings.filterwarnings(action='ignore')

from beartype import beartype
import traceback
import logging
import torch
import time
import copy
import os


from heuristic.restart_heuristics import HIDDEN_SPLIT_RESTART_STRATEGIES, INPUT_SPLIT_RESTART_STRATEGIES
from heuristic.domains_list import DomainsList

from verifier.utils import _prune_domains, get_used_gpu_memory
from verifier.mip_solver import MIPSolver

from abstractor.auto_LiRPA.utils import stop_criterion_batch_any
from abstractor.utils import new_slopes

from helper.misc.torch_cuda_memory import is_cuda_out_of_memory, gc_cuda
from helper.misc.error import VerifierInitializeError
from helper.network.onnx2pytorch import ConvertModel
from helper.spec.objective import DnfObjectives
from helper.misc.result import ReturnStatus
from helper.misc.logger import logger

from setting import Settings


class Verifier:
    
    "Branch-and-Bound verifier"
    
    @beartype
    def __init__(self: 'Verifier', net: ConvertModel | torch.nn.Module , input_shape: tuple, batch: int = 1000, device: str = 'cpu') -> None:
        self.net = net # pytorch model
        self.input_shape = input_shape
        self.device = device
        
        # hyper parameters
        self.input_split = False
        self.batch = max(batch, 1)
        self.orig_batch = max(batch, 1)

        # counter-example
        self.adv = None
        
        # debug
        self.iteration = 0
        self.last_minimum_lowers = -1e9
        self.tightening_patience = 0
        
        # stats
        self.all_conflict_clauses = {}
        self.visited = 0
        
        # other verifier
        self.other = copy.deepcopy(self)

        self.falsified = 0
        self.certified = 0
        self.undecided = 0

    @beartype
    def get_objective(self: 'Verifier', dnf_objectives: 'DnfObjectives', max_domain: int):
        # objective = dnf_objectives.pop(1)
        objective = dnf_objectives.pop(max(1, max_domain))
        return objective
    
    @beartype
    def verify(self: 'Verifier', dnf_objectives: 'DnfObjectives', preconditions: list = [], timeout: int | float = 3600.0, force_split: str | None = None) -> dict:
        self.start_time = time.time()
        self.total_time = timeout
        self.status = self._verify(
            dnf_objectives=dnf_objectives,
            preconditions=preconditions,
            timeout=timeout,
            force_split=force_split,
        )
        for obj_id in self.status:
            falsified = round(self.status[obj_id]["falsified"], 4) * 100
            proved = round(self.status[obj_id]["proved"], 4) * 100
            undecided = round(100 - falsified - proved, 2)
            self.status[obj_id] = {
                "falsified": falsified,
                "proved": proved,
                "undecided": undecided
            }
        return self.status
    
    
    @beartype
    def _verify(self: 'Verifier', dnf_objectives: 'DnfObjectives', preconditions: list, timeout: int | float = 3600.0, force_split: str | None = None) -> dict:
        """
        Return a quantitive verification result

        Returns:
            A tuple of Certified %, Falsified %, Undecided %
        """
        if not len(dnf_objectives):
            return 100.0, 0.0, 0.0

        # Ensure it's always input_split
        assert force_split == "input"

        # refine
        dnf_objectives, reference_bounds = self._preprocess(dnf_objectives, force_split=force_split)
        
        if os.environ.get('NEURALSAT_DEBUG'):
            print(f'[+] verify _preprocess:', get_used_gpu_memory(), 'MB')
            
        if not len(dnf_objectives):
            return ReturnStatus.UNSAT

        # FIXME: generalize this
        max_domain = min(self.batch, len(dnf_objectives))
        status = self._verify_with_restart(
            dnf_objectives=copy.deepcopy(dnf_objectives),
            preconditions=preconditions,
            timeout=timeout,
            reference_bounds=reference_bounds,
            max_domain=max_domain
        )
            
        return status
        
    def _heuristic_configure(self: 'Verifier', timeout: int | float) -> None:
        if timeout <= 30:
            Settings.use_restart = False
            Settings.use_mip_tightening = False
        
        if timeout <= 60:
            Settings.restart_max_runtime_percentage = 0.7
            
    @beartype    
    def _verify_with_restart(self: 'Verifier', dnf_objectives: 'DnfObjectives', preconditions: list, 
                             timeout: int | float = 3600.0, reference_bounds: None | dict = None, max_domain: int = 1) -> dict:
        # verify
        while len(dnf_objectives):
            objective = self.get_objective(dnf_objectives, max_domain=max_domain)
            
            # restart variables
            nth_restart = 0 
            learned_clauses = {int(k): [] for k in objective.ids}
            # TODO: shouldn't add to all objectives
            if len(preconditions): # add to all objective ids
                [learned_clauses[k].extend(preconditions) for k in learned_clauses]
            
            self._heuristic_configure(timeout=timeout)
            
            # verify objective (multiple times if RESTART is returned)
            while True:
                # get strategy + refinement
                new_reference_bounds = self._setup_restart(nth_restart, objective)
                
                # adaptive batch size
                while True: 
                    try:
                        # main function
                        logger.info(f'Try batch size {self.batch}')
                        status = self._verify_one(
                            objective=objective, 
                            preconditions=learned_clauses, 
                            reference_bounds=reference_bounds if new_reference_bounds is None else new_reference_bounds,
                            timeout=timeout
                        )
                    except RuntimeError as exception:
                        if os.environ.get("NEURALSAT_DEBUG"):
                            traceback.print_exc()
                            raise NotImplementedError('Unsupported exception')
                        
                        if is_cuda_out_of_memory(exception):
                            if self.batch == 1:
                                # cannot find a suitable batch size to fit this device
                                logger.debug('[!] OOM with batch_size=1')
                                return ReturnStatus.UNKNOWN
                            self.batch = self.batch // 2
                            dnf_objectives.add(objective)
                            objective = self.get_objective(dnf_objectives, max_domain=max_domain)
                            continue
                        else:
                            logger.debug('[!] RuntimeError exception')
                            traceback.print_exc()
                            return None
                    except SystemExit:
                        exit()
                    except:
                        raise NotImplementedError('Unknown error')
                    else:
                        gc_cuda()
                        break
                    
                # stats
                self._save_stats()
                return status
                
                # handle returning status
                if status in [ReturnStatus.SAT, ReturnStatus.TIMEOUT, ReturnStatus.UNKNOWN, ReturnStatus.EARLY_STOP]:
                    return status 
                if status == ReturnStatus.UNSAT:
                    break # objective is verified
                if status == ReturnStatus.RESTART:
                    logger.debug('Restarting')
                    # restore original batch size for new restart
                    objective = self._prune_objective(objective)
                    self.batch = self.orig_batch
                    nth_restart += 1
                    # TODO: check general activation
                    if not self.input_split:
                        for k, v in self._get_learned_conflict_clauses().items():
                            learned_clauses[k].extend(v)
                    continue
                raise NotImplementedError(status)
            
            logger.info(f'Verified: {len(objective.cs)} \t Remain: {len(dnf_objectives)}')
            
        return ReturnStatus.UNSAT  
                
        
    @beartype
    def _initialize(self: 'Verifier', objective, preconditions: dict, reference_bounds: dict | None) -> DomainsList | list:
        # initialization params
        # TODO: fix init_betas found by MIP
        ret = self.abstractor.initialize(objective, reference_bounds=reference_bounds)

        # check verified
        assert len(ret.output_lbs) == len(objective.cs)
        if stop_criterion_batch_any(objective.rhs.to(self.device))(ret.output_lbs.to(self.device)).all():
            return []

        self.result = dict()
        for idx in ret.objective_ids:
            self.result[idx.item()] = {
                "falsified": 0, "proved": 0
            }

        # full slopes uses too much memory
        slopes = ret.slopes if self.input_split else new_slopes(ret.slopes, self.abstractor.net.final_name)
        
        # remaining domains
        return DomainsList(
            net=self.abstractor.net,
            objective_ids=ret.objective_ids,
            output_lbs=ret.output_lbs,
            output_ubs=ret.output_ubs,
            input_lowers=ret.input_lowers,
            input_uppers=ret.input_uppers,
            lower_bounds=ret.lower_bounds, 
            upper_bounds=ret.upper_bounds, 
            lAs=ret.lAs, 
            slopes=slopes, # pruned slopes
            histories=copy.deepcopy(ret.histories), 
            cs=ret.cs,
            rhs=ret.rhs,
            input_split=self.input_split,
            preconditions=preconditions,
        )
        
        
    @beartype
    def _verify_one(self: 'Verifier', objective, preconditions: dict, reference_bounds: dict | None, timeout: int | float) -> dict:
        # initialization
        try:
            self.domains_list = self._initialize(objective=objective, preconditions=preconditions, reference_bounds=reference_bounds)
        except RuntimeError as exception:
            if is_cuda_out_of_memory(exception):
                raise VerifierInitializeError('[_verify_one] OOM exception')
            else:
                raise VerifierInitializeError('[_verify_one] Unknown exception')
        except:
            raise VerifierInitializeError('[_verify_one] Unknown error')
        
        if os.environ.get('NEURALSAT_DEBUG'):
            print(f'[+] verify _initialize:', get_used_gpu_memory(), 'MB')
                
        # cleaning
        torch.cuda.empty_cache()
        if hasattr(self, 'milp_tightener'):
            self.milp_tightener.reset()
            
        if hasattr(self, 'gpu_tightener'):
            self.gpu_tightener.reset()
        
        # main loop
        start_time = time.time()
        start_iteration = self.iteration

        while len(self.domains_list) > 0:
            # early stop
            if self.domains_list.minimum_lowers < Settings.skip_initial_worst_bound:
                return ReturnStatus.EARLY_STOP
            
            # search
            self._parallel_dpll()
                
            # check adv founded
            if self.adv is not None:
                if self._check_adv(self.adv, objective):
                    return ReturnStatus.SAT
                logger.debug("[!] Invalid counter-example")
                # FIXME
                return ReturnStatus.INVALID_CEX
                self.adv = None
            
            # check timeout
            if self._check_timeout(timeout):
                return ReturnStatus.TIMEOUT
            

            # check unsolvable
            if len(self.domains_list) > Settings.max_domains:
                return ReturnStatus.UNKNOWN
            
            # gpu tightening early stop
            if self._stop_gpu_tightening():
                return ReturnStatus.UNKNOWN
            
            # early stop
            if self.iteration >= Settings.max_iterations:
                return ReturnStatus.EARLY_STOP
            
        return self.result

    
    @beartype
    def _stop_gpu_tightening(self) -> bool:
        if hasattr(self, 'other'): # main verifier:
            return False
        
        if not Settings.use_gpu_tightening:
            return False
        
        if len(self.domains_list) > Settings.gpu_tightening_current_hidden_branches:
            return True
        
        if self.domains_list.visited > Settings.gpu_tightening_visited_hidden_branches:
            return True
        
        return False
        
            
            
    @beartype
    def _parallel_dpll(self: 'Verifier') -> None:
        iter_start = time.time()
        
        # step 1: MIP attack
        if Settings.use_mip_attack:
            self.mip_attacker.attack_domains(self.domains_list.pick_out_worst_domains(1001, 'cpu'))
        
        # step 2: stabilizing
        old_domains_length = len(self.domains_list)
        unstable = self.domains_list.count_unstable_neurons()
        if self._check_invoke_cpu_tightening(patience_limit=Settings.mip_tightening_patience):
            self.milp_tightener(
                domain_list=self.domains_list, 
                topk=Settings.mip_tightening_topk, 
                timeout=Settings.mip_tightening_timeout_per_neuron, 
                largest=False, # stabilize near-stable neurons
            )
            
        if self._check_invoke_gpu_tightening(patience_limit=Settings.gpu_tightening_patience):
            self.gpu_tightener(
                domain_list=self.domains_list, 
                topk=Settings.gpu_tightening_topk, 
                iteration=2,
            )
            
        # step 3: selection
        tic = time.time()
        pick_ret = self.domains_list.pick_out(self.batch, self.device)
        pick_time = time.time() - tic
        
        # step 4: PGD attack
        tic = time.time()
        self.adv = self._attack(pick_ret, n_interval=Settings.attack_interval, timeout=1.0)
        attack_time = time.time() - tic
        if self.adv is not None:
            return

        # step 5: complete assignments
        # If input split, always return None, None
        # self.adv, remain_idx = self._check_full_assignment(pick_ret)
        # if (self.adv is not None): 
        #     return
        
        # pruning/ filter
        # remain_idx is always None for input split, so pick_ret
        pruned_ret = pick_ret
        if not len(pruned_ret.input_lowers): 
            return
            
        # step 6: branching
        tic = time.time()
        decisions = self.decision(self.abstractor, pruned_ret)
        decision_time = time.time() - tic
        
        # step 7: abstraction 
        tic = time.time()
        abstraction_ret = self.abstractor.forward(decisions, pruned_ret)
        abstraction_time = time.time() - tic

        # step 8: pruning unverified branches
        tic = time.time()
        proved_idx, falsified_idx = self.domains_list.add(abstraction_ret, decisions)

        for idx in proved_idx:
            obj_id = int(abstraction_ret.objective_ids[idx].item())
            depth  = int(abstraction_ret.split_depth[idx].item())
            self.result[obj_id]["proved"] += 1 / (2 ** depth)

        for idx in falsified_idx:
            obj_id = int(abstraction_ret.objective_ids[idx].item())
            depth  = int(abstraction_ret.split_depth[idx].item())
            self.result[obj_id]["falsified"] += 1 / (2 ** depth)

        add_time = time.time() - tic

        # statistics
        self.iteration += 1
        minimum_lowers = self.domains_list.minimum_lowers
        self._update_tightening_patience(minimum_lowers, old_domains_length)
        
        # adapt batch size
        current_batch = len(pick_ret.input_lowers)
        _, mem_used_percentage = get_used_gpu_memory(return_percentage=True)
        if mem_used_percentage > 80.0:
            self.batch = current_batch
            logger.debug(f'Fixed {self.batch=}')
        elif self.input_split and (current_batch < old_domains_length) and (self.num_restart < len(INPUT_SPLIT_RESTART_STRATEGIES)) and (self.abstractor.method != 'crown-optimized'):
            if mem_used_percentage < 10.0:
                self.batch = min(500000, self.batch*10)
                logger.debug(f'Increase {current_batch=} {old_domains_length=} {self.batch=}')
            elif mem_used_percentage < 50.0:
                self.batch = min(500000, self.batch*2)
                logger.debug(f'Increase {current_batch=} {old_domains_length=} {self.batch=}')
            
        # logging
        msg = (
            f'[{"Input" if self.input_split else "Hidden"} splitting]     '
            f'Iteration: {self.iteration:<10} '
            f'Remaining: {len(self.domains_list):<10} '
            f'Visited: {self.domains_list.visited:<10} '
            f'Bound: {minimum_lowers:<15.06f} '
            f'Time elapsed (s): {time.time() - self.start_time:<10.02f} '
        )
        if logger.level <= logging.DEBUG:
            msg += f'Iteration elapsed (s): {time.time() - iter_start:<10.02f} '
            
            if Settings.use_mip_tightening and (not self.input_split):
                msg += f'CPU Tightening patience: {self.tightening_patience}/{Settings.mip_tightening_patience:<10}'
                
            if Settings.use_gpu_tightening and (not self.input_split):
                msg += f'GPU Tightening patience: {self.tightening_patience}/{Settings.gpu_tightening_patience:<10}'
                
            if (not self.input_split) and (unstable is not None):
                msg += f'Unstable neurons: {unstable:<10}'
            
            msg += f'GPU Mem (%): {mem_used_percentage:<10.02f}'
            msg += f'Batch: {self.batch:<10}'
            msg += f'Restart: {self.num_restart:<10}'
            
        logger.info(msg)
        
            
        if os.environ.get("NEURALSAT_TIMING"):
            # DEBUG: sometimes add_time could be very high
            logger.debug(f'[TIMING] {pick_time=:<10.03f} {attack_time=:<10.03f} {decision_time=:<10.03f} {abstraction_time=:<10.03f} {add_time=:<10.03f}')
        
    
    
    from .utils import (
        _preprocess, 
        _init_abstractor,
        _check_timeout,
        _setup_restart, _setup_restart_naive,
        _pre_attack, _attack, _mip_attack, _check_adv,
        _get_learned_conflict_clauses, _check_full_assignment,
        _check_invoke_cpu_tightening, _update_tightening_patience,
        _check_invoke_gpu_tightening,
        _save_stats, get_stats,
        _prune_objective,
        get_proof_tree, export_proof,
        _check_invoke_mip_presolving,
    )
    