import pandas as pd
import numpy as np
import clingo

from robustx.datasets.DatasetLoader import DatasetLoader
from robustx.generators.CEGenerator import CEGenerator
from robustx.generators.CE_methods.NNCE import NNCE
from robustx.lib.tasks.ClassificationTask import ClassificationTask

BAF_ENCODING = """
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Encodings for BAFs
% to compute: d-admissible,
%    	      c-admissible,
%	      s-admissible,
%	      d-preferred,
%	      c-preferred and
%	      s-preferred extensions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

adm :- d_adm, baf, not input_error.		% d-admissible extensions for BAF's
						%  are the same as the standard admissible extensions
adm :- prefex, not baf, not input_error.
comp :- ground, not input_error.		% every grounded ext. is also a complete ext.
prefex :- d_prefex, baf, not input_error.	% d-preferred ext. for BAF's are the same as
			 			% standard pref. ext.
d_adm :- d_prefex, baf, not input_error.
closed :- c_adm, baf, not input_error.		% c-adm. ext. for BAF's need to be closed
safe :- s_adm, not input_error.			% s-adm. ext. for BAF's need to be safe
s_adm :- s_prefex, baf, not input_error.
c_adm :- c_prefex, baf, not input_error.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	support and defeat for BAF
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% argument x is supported by argument y
support(X,Z) :- support(X,Y), support(Y,Z).

%% set-supports: argument x is supported by the set S
supported(X) :- support(Y,X), in(Y).

%% defeats (BAF)
defeat(X,Y) :- att(Z,Y), support(X,Z), baf. %supported defeat
defeat(X,Y) :- att(X,Y), baf.	     %supported defeat
defeat(X,Y) :- att(X,Z), support(Z,Y), baf. %indirekt defeat
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% an argument x defeats an argument y if x attacks y
defeat(X,Y) :- att(X,Y).

%% Guess a set S subseteq A
in(X) :- not out(X), arg(X).
out(X) :- not in(X), arg(X).

%% S has to be conflict-free
:- in(X), in(Y), defeat(X,Y).

%% The argument x is defeated by the set S
defeated(X) :- in(Y), defeat(Y,X).

%% The argument x is not defended by S
not_defended(X) :- defeat(Y,X), not defeated(Y).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% speciall semantics for BAF
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% safe
:- supported(B), defeated(B), safe.
:- defeated(B), in(B), safe.

%% s-admissible
:- in(X), not_defended(X), s_adm.

%% closed
:- support(X,Y), out(Y),in(X), closed.
:- support(X,Y), in(Y), out(X), closed.

%% c_admissible
:- in(X), not_defended(X), c_adm.

%% d_admissible
:- in(X), not_defended(X), d_adm.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For the remaining part we need to put an order on the domain.
% Therefore, we define a successor-relation with infinum and supremum
% as follows
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

lt(X,Y) :- arg(X),arg(Y), X<Y, not input_error.
nsucc(X,Z) :- lt(X,Y), lt(Y,Z).
succ(X,Y) :- lt(X,Y), not nsucc(X,Y).
ninf(X) :- lt(Y,X).
nsup(X) :- lt(X,Y).
inf(X) :- not ninf(X), arg(X).
sup(X) :- not nsup(X), arg(X).

%% Guess S' supseteq S for classical pref. ext.

inN(X) | outN(X) :- out(X), prefex, not input_error.
inN(X) :- in(X), prefex, not input_error.

%% Guess S' supseteq S for s-preferred
inN(X) | outN(X) :- out(X), s_prefex, not input_error.
inN(X) :- in(X), s_prefex.

%% Guess S' supseteq S for c-preferred
inN(X) | outN(X) :- out(X), c_prefex, not input_error.
inN(X) :- in(X), c_prefex.

%% Guess S' supseteq S for d-preferred
inN(X) | outN(X) :- out(X), d_prefex, not input_error.
inN(X) :- in(X), d_prefex.


%% If S' = S then spoil.
%% Use the sucessor function and check starting from supremum whether
%% elements in S' is also in S. If this is not the case we "stop"
%% If we reach the supremum we spoil up.

% eq indicates whether a guess for S' is equal to the guess for S
eq_upto(Y) :- inf(Y), in(Y), inN(Y), not semis.
eq_upto(Y) :- inf(Y), out(Y), outN(Y), not semis.

eq_upto(Y) :- succ(Z,Y), in(Y), inN(Y), eq_upto(Z).
eq_upto(Y) :- succ(Z,Y), out(Y), outN(Y), eq_upto(Z).

eq :- sup(Y), eq_upto(Y).

%% get those X not in S' which are not defeated by S'
%% using successor again...

undefeated_upto(X,Y) :- inf(Y), outN(X), outN(Y), prefex.
undefeated_upto(X,Y) :- inf(Y), outN(X),  not defeat(Y,X), prefex.

undefeated_upto(X,Y) :- inf(Y), outN(X), outN(Y), s_prefex.
undefeated_upto(X,Y) :- inf(Y), outN(X),  not defeat(Y,X), s_prefex.

undefeated_upto(X,Y) :- inf(Y), outN(X), outN(Y), c_prefex.
undefeated_upto(X,Y) :- inf(Y), outN(X),  not defeat(Y,X), c_prefex.

undefeated_upto(X,Y) :- inf(Y), outN(X), outN(Y), d_prefex.
undefeated_upto(X,Y) :- inf(Y), outN(X),  not defeat(Y,X), d_prefex.

undefeated_upto(X,Y) :- inf(Y), outN(X), outN(Y), semis.
undefeated_upto(X,Y) :- inf(Y), outN(X),  not defeat(Y,X), semis.

undefeated_upto(X,Y) :- succ(Z,Y), undefeated_upto(X,Z), outN(Y).
undefeated_upto(X,Y) :- succ(Z,Y), undefeated_upto(X,Z), not defeat(Y,X).

undefeated(X) :- sup(Y), undefeated_upto(X,Y).

%% spoil if S' equals S for all preferred extensions
spoil :- eq.

%% S' has to be conflictfree - otherwise spoil
spoil :- inN(X), inN(Y), defeat(X,Y), c_prefex.
spoil :- inN(X), inN(Y), defeat(X,Y), d_prefex.
spoil :- inN(X), inN(Y), defeat(X,Y), prefex.

%% set-supports
supportedN(X) :- support(Y,X), inN(Y).

%% S' has to be safe for s-preferred
spoil :- supportedN(B), defeat(X,B), inN(X), s_prefex.
spoil :- defeat(X,B), inN(X), inN(B), s_prefex.

%% S' has to be closed for c-preferred
spoil :- support(X,Y), outN(Y), inN(X), c_prefex.
spoil :- support(X,Y), inN(Y), outN(X), c_prefex.

%% S' has to be admissible - otherwise spoil
spoil :- inN(X), outN(Y), defeat(Y,X), undefeated(Y).

inN(X) :- spoil, arg(X), not input_error.
outN(X) :- spoil, arg(X), not input_error.

%% do the final spoil-thing ...
:- not spoil, prefex.
:- not spoil, s_prefex.
:- not spoil, c_prefex.
:- not spoil, d_prefex.

#show in/1.

"""


class ArgEnsembling(CEGenerator):
    """
    A counterfactual explanation generator that deals with the model multiplicity / predictive multiplicity scenario. Use computational
    argumentation to resolve conflicts, select the models which agree on classification results,
    and whose counterfactuals are valid on the other models in the set.

    This is a simplified version of Jiang et al. AAMAS 2024 paper, not implementing preferences over models.

    Attributes:
        _task (Task): The task to solve, inherited from CEGenerator.
        models: The set of models with which the model multiplicity problem is instantiated
        dl: The dataset
    """

    def __init__(self, dl: DatasetLoader, models):
        """
        Initializes the Argumentative Ensembling CE generator with a dataset and a list of models.

        Args:
            dl: dataset loader
            models: the list of models forming the model multiplicity problem setting
        """
        super().__init__(ClassificationTask(models[0], dl))
        self.models = models
        self.dl = dl

    def _generation_method(self, instance, **kwargs) -> pd.DataFrame:
        """
        Generate CE for the instance with a set of models possibly giving conflicting predictions.
        Can return one or more counterfactual explanation points

        Args:
            instance: The input point for which CE is generated
            **kwargs: Additional keyword arguments

        Returns:
            One or more counterfactual explanation points
        """
        # get model predictions and counterfactuals
        x = instance.values
        # models predictions on the input
        res = []
        ces = np.zeros((len(self.models), len(x)))

        for i, m in enumerate(self.models):
            pred = m.predict_single(pd.DataFrame(x.reshape(1, -1)))
            res.append(pred)
            ce_gen = NNCE(ClassificationTask(m, self.dl))
            ce = ce_gen.generate_for_instance(pd.DataFrame(x.reshape(1, -1)), neg_value=pred)
            ces[i] = ce.values.flatten()[:len(x)]

        # get counterfactual predictions
        ces_pred = np.zeros((len(self.models), len(self.models)))
        for i, m in enumerate(self.models):
            for j, c in enumerate(ces):
                ces_pred[i][j] = m.predict_single(pd.DataFrame(c.reshape(1, -1)))

        # write bipolar argumentation framework for this input
        baf = "baf.\n"
        baf += "s_prefex.\n"

        # add arguments
        for i, m in enumerate(self.models):
            baf += f"arg(m{i}).\n"
            baf += f"arg(c{i}).\n"

        # add attacks and supports:
        for i, m1 in enumerate(self.models):
            baf += f"support(m{i},c{i}).\n"
            baf += f"support(c{i},m{i}).\n"
            for j, m2 in enumerate(self.models):
                if res[j] == ces_pred[j][i]:
                    baf += f"att(m{j},c{i}).\n"
                    baf += f"att(c{i},m{j}).\n"
                if res[i] != res[j]:
                    if j < i:
                        continue
                    baf += f"att(m{j},m{i}).\n"
                    baf += f"att(m{i},m{j}).\n"

        # solve
        baf_to_solve = BAF_ENCODING + baf

        # Clingo Control Object for solving BAF
        ctl = clingo.Control()
        ctl.add("base", [], baf_to_solve)
        ctl.ground([("base", [])])
        ctl.configuration.solve.models = "0"
        exts = []
        with ctl.solve(yield_=True) as handle:
            for model in handle:
                exts.append(self._get_extension(model))

        # get indices of models and counterfactuals
        largest_set_idx = 0
        for i, lst in enumerate(exts):
            if len(lst) >= len(exts[largest_set_idx]):
                largest_set_idx = i
        optimal_idxs = exts[largest_set_idx]

        # return counterfactuals only
        return pd.DataFrame(ces[optimal_idxs])

    def _get_extension(self, m):
        """
        Helper function to retrieve accepted arguments from the clingo solver.

        Args:
            m: The extension of the BAF, in the form of clingo solved model.

        Returns:
            The list of indices of accepted model (and counterfactual) in one extension set of the BAF
        """
        atoms = [str(atom.arguments[0]) for atom in m.symbols(atoms=True) if atom.name == "in"]
        atoms_idx = []
        for item in atoms:
            idx = int(item[1:])
            if idx not in atoms_idx:
                atoms_idx.append(idx)
        return atoms_idx
