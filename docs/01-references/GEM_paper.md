<figure id="fig:all_baselines">
<embed src="imgs/24envs.pdf" />
<figcaption>Learning curves of Qwen3-based agents across diverse
environments of 5 categories: <span
style="color: game_color"><strong>game</strong></span> (language games);
<span style="color: rg_color"><strong>rg</strong></span> (ReasoningGym);
<span style="color: code_color"><strong>code</strong></span> (coding
tasks); <span style="color: math_color"><strong>math</strong></span>
(python-integrated math questions); <span
style="color: qa_color"><strong>qa</strong></span> (search-integrated
general questions). All agents are learned via a simple yet general
multi-turn algorithm based on REINFORCE (). The comparison between two
curves in each subplot illustrate the effectiveness of Return Batch
Normalization (ReBN).</figcaption>
</figure>

# Introduction

Reinforcement learning (RL) (Sutton and Barto 2018) has emerged as a
powerful paradigm for improving the reasoning capabilities of large
language models (LLMs) (OpenAI 2024; Guo et al. 2025). By collecting
*experience* in interactive environments, RL allows agents to learn
complex, open-ended tasks without direct supervision (Silver and Sutton
2025). This approach promises to create powerful agents for a variety of
domains. For instance, an agent could develop entire software modules by
writing, testing, and debugging code, while also adapting to integration
failures or evolving requirements. Similarly, in scientific discovery,
an agent could be trained to develop hypotheses, design relevant
experiments, and adjust its long-term strategy based on the results.

However, current research on RL for LLMs has largely focused on
single-turn tasks, such as answering math questions or retrieving
specific data (Lambert et al. 2024; Guo et al. 2025). While these tasks
are a valuable starting point, they significantly oversimplify
multi-turn interactions (B. Liu et al. 2025). This oversimplification
means that algorithms which excel in the single-turn setting (e.g.,
GRPO (Shao et al. 2024)) are fundamentally inapplicable to full
multi-turn problems. If the goal is to train agentic LLMs capable of
long-horizon planning, trial-and-error, iterative refinement etc, it is
crucial to transition to testbeds that support these more complex
multi-turn interactions.

To facilitate this next step, we introduce GEM (General Experience
Maker), an open-source environment framework for diverse, multi-turn,
long-horizon tasks. Motivated by OpenAI-Gym (Brockman et al. 2016) which
catalyzed research in traditional RL by providing a unified interface
and standardized environments, GEM aims to provide analogous
foundational infrastructure for LLM agents. GEM offers a diverse suite
of environments spanning single- and multi-turn (over $100$ turns) tasks
(including tool integrated responses, reasoning games etc), flexible
observation and action wrappers, asynchronous parallel execution, and a
rich set of tools (python, search, and external MCP compatible tools).
Additionally, GEM includes validated baselines and single-file training
scripts showcasing seamless integration with five popular RL training
frameworks (Oat, Verl, OpenRLHF, ROLL, and RL2—see ).

Besides introducing the GEM framework, this paper also presents and
discusses a simple yet effective algorithmic variant of
REINFORCE (Williams 1992) which incorporates *Return Batch
Normalization* (ReBN), a useful technique similar to advantage
normalization (Andrychowicz et al. 2021; Z. Liu et al. 2025) that brings
consistent improvements (). Unlike GRPO and its variants, REINFORCE with
ReBN is fully compatible with the multi-turn RL setting, including
turn-level dense rewards and arbitrary discount factors. We further
compare REINFORCE-based algorithms with multi-turn PPO (Schulman et al.
2017) and GRPO, showing its theoretical connections and empirical
tradeoffs. We also provide case studies on the impact of the discount
factor $\gamma$ on multi-turn learning, extensive results of
tool-integrated RL, and performance benchmarks on terminal and MCP usage
of strong LLMs using GEM as a unified evaluation toolkit. We hope this
framework will accelerate RL research on agentic LLMs and advance
progress toward more capable and autonomous AI systems.

# GEM environments

This section introduces GEM’s core functionality, covering its main
interface (), the environment design (), and advanced features such as
asynchronous vectorization and modular wrappers ().

## Interface

<span id="sec:interface" label="sec:interface"></span>

GEM employs a standardized environment interface closely following the
well-established OpenAI Gym API with the main functions being `reset()`
and `step()`. A basic agent-environment interaction loop is as follows
(multi-agent interface shown in ):

``` python
import gem
# gem.print_envs() # to list all available environments
env = gem.make("game:GuessTheNumber-v0")
observation, info = env.reset()

while True:
    # (1) Agent acting:
    action = env.sample_random_action()
    # action = agent.act(observation) # real acting by LLM sampling

    # (2) Environment execution:
    next_obs, reward, terminated, truncated, info = env.step(action)

    # (3) Agent learning:
    # agent.learn(observation, action, reward)

    observation = next_obs
    if terminated or truncated: break
```

## Tasks and tools

<span id="sec:tasks_and_tools" label="sec:tasks_and_tools"></span>

GEM’s core environment components are **tasks** and **tools**. Each
combination of a task and an optional set of tools constitutes an
environment that tests complex capabilities such as reasoning,
multi-step planning, and tool use. These environments can therefore be
used to benchmark LLMs and to test and develop new algorithms. GEM
currently features seven main categories of tasks:

<div class="tcolorbox">

**Math:** Solve math problems with chain-of-thought reasoning.

**Math with image:** Solve geometry math problems with images using
chain-of-thought reasoning.

**Code:** Generate code to solve competitive programming problems.

**Game:** Multi-turn text-based games adapted from TextArena (Guertler
et al. 2025).

**QA:** General, potentially knowledge-intensive questions (useful for
testing search tool capability).

**ReasoningGym:** A unified interface of ReasoningGym (Stojanovski et
al. 2025) which provides $100+$ single-turn verifiable tasks.

**Terminal**: Perform complex tasks through a containerized terminal
environment.

</div>

GEM’s modular design simplifies task integration. Math (with images),
code, and QA tasks can be integrated by simply providing a new dataset.
Terminal tasks require a new Docker file, instructions, and test cases.
New games and other custom tasks can be added by inheriting from GEM’s
environment base class and defining their state transition and reward
logic. In addition, tasks can be augmented with any combination of
tools. GEM currently supports:

<div class="tcolorbox">

**Python:** Parses and executes code blocks, returning the stdout or
execution error.

**Search:** Parses a query, executes a search against an external
engine, and returns the results.

**MCP:** General tool calling to any external servers that conform to
the model context protocol.

</div>

The use of tools converts single-turn tasks, like Math or ReasoningGym,
into multi-turn tasks in which an agent can learn to call tools and
adapt based on their output.

## Asynchronous vectorization and autoreset

<span id="sec:async" label="sec:async"></span>

To facilitate efficient agent RL training, we support parallel execution
of vectorized environments via asynchronous tool calls to collect
episodes in batches. In addition to the latency reduction, the use of
vectorized environments with autoreset streamlines the experience
collection logic. Users can run a single `.reset()` at the
initialization stage and simply continue with `.step()` in the following
agent-environment loop for continuous data generation. In addition, the
user code can use the returned `terminated` flag to prevent value
bootstrapping across episode boundaries, ensuring the correctness of
critic learning. An illustration of the autoreset mechanism can be found
in .

<figure id="fig:parallel_env">
<embed src="imgs/parallel_env.pdf" />
<figcaption>Illustration of autoreset in vectorized environments.
Autoresetting resets the environment automatically after termination,
allowing users to collect batches of episodes by simply running
<code>.step()</code> without needing more complicated logic such as
keeping track of whether individual episodes have
terminated.</figcaption>
</figure>

## Wrappers

Like in OpenAI-Gym, GEM uses wrappers for easy extensibility.
Observation wrappers, for example, control how the episode is converted
into an observation. Options include observing just the most recent
environment output, a concatenation of all previous environment outputs,
a concatenation of all previous environment outputs and actions, or some
parsed/summarized version of this. The Python interpreter or
database/web search tools are also formulated as wrappers which can be
added on top of any specified task environment.

# Reinforcement learning with GEM

In this section, we begin by describing the main RL formulations for
LLMs, including their respective flexibilities and limitations ().
Motivated by this, we then present our baseline algorithm which is
applicable to the more flexible RL formulation ().

<figure id="fig:llm_agent_formulation">
<embed src="imgs/llm_agent_formulation.pdf" />
<figcaption>The illustration of different view of agentic RL. Green
nodes denote tokens responsible for loss.</figcaption>
</figure>

## Preliminary: LLMs as agents

<span id="sec:llm_as_agents" label="sec:llm_as_agents"></span> There are
three main ways of treating LLM-environment interactions in RL
algorithms which each have different limitations and strengths:

**Action = Single token ((a)):** The first approach is to treat each
token generated by the LLM as an individual action (Ziegler et al.
2019). This, however, means that episodes are typically very long
(thousands of tokens), and it also requires specifying the reward for
the addition of every token, which is difficult to evaluate. Successful
applications of RL in this formulation tend to use sparse outcome reward
with discount factor $\gamma=1$ (Guo et al. 2025).

**Action = Response ((b)):** To avoid these complications the second
approach is to treat a whole response (a sequence of tokens until an
EOS) as a single action[^3] (Ahmadian et al. 2024; B. Liu et al. 2025).
In answering math problems for example—currently the most common testbed
for RL for LLMs—each episode contains a question and response. With this
view all episodes therefore have length 1 and the RL problem essentially
degenerates to contextual bandits (Abe, Biermann, and Long 2003). This
is convenient as it means sample-based advantage estimation methods such
as GRPO (Shao et al. 2024) can be applied efficiently, and these have
been demonstrated to be highly effective. Extending to multi-turn
episodes (e.g. for games or tool use), however, results in an issue:
Multi-turn interactions have episode lengths $>1$, meaning sample-based
advantage estimation methods (e.g., Kazemnejad et al. (2025)) become
infeasible (since they require collecting multiple episode completions
from each turn (state) in the episode, leading to exponential
complexity).

**Action = Whole interaction ((c)):** One approach to make GRPO
applicable to multi-turn interactions is to treat the whole interaction
as a single action while masking the loss on tool outputs. This view
again degenerates the full RL problem back to one-step RL or contextual
bandits, meaning GRPO etc. can be applied. However, it requires two
compromises: Firstly, it effectively fixes the discount factor at
$\gamma=1$, thus removing the incentive to solve problems quickly. This
is significant, for example in where we show how the optimal search
algorithm is only recovered when setting $\gamma<1$. Secondly, this
approach is limited to single trajectory-level rewards, losing
fine-grained per-turn credit assignment.

Many prior works make these concessions and use GRPO in multi-turn LLM
RL (Cao et al. 2025; Jiang et al. 2025; M. Chen et al. 2025; Jin et al.
2025; J. Feng et al. 2025). However, to develop an algorithm compatible
with the full RL setting, we go back to the second view
(action=response) and employ a simple variant of REINFORCE with *Return
Batch Normalization* (ReBN). Unlike GRPO, this algorithm is compatible
with per-step dense rewards and arbitrary discount factors
($\gamma\leq1$), thus making it significantly more flexible for
optimizing LLMs in complex, multi-turn interactive settings.

## Baseline algorithms

<span id="sec:baseline_algo" label="sec:baseline_algo"></span>

We start from the foundational on-policy[^4] policy-gradient method
REINFORCE (Williams 1992), which optimizes the following objective:
$$\label{eq:reinforce}
    \mathcal{J}_{\text{REINFORCE}}(\theta) = \frac{1}{N} \sum_{n=1}^{N} \sum_{t=0}^{T^{(n)}-1} G^{(n)}_t \log\pi_{\theta}(a^{(n)}_t|s^{(n)}_t),
\vspace{-2mm}$$ where $N$ is the batch size,
$[s_0, a_0, s_1, ..., a_{T-1}]$ is a sequence of states and actions
making up a trajectory in which each $s_t$ and $a_t$ is itself a
sequence of tokens, and $G_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_k$ is the
return. Though initially designed for single-turn problems (i.e.,
$T^{(n)}=1$), GRPO can be extended to multi-turn tasks by sampling a
group of $M$ trajectories per initial state and normalizing the
trajectory-level reward for each group[^5]: $$\label{eq:grpo}
    \mathcal{J}_{\text{GRPO}}(\theta) = \frac{1}{N} \sum_{n=1}^{N} \frac{1}{M} \sum_{m=1}^{M} A_{\text{GRPO}}^{(n,m)} \sum_{t=0}^{T^{(n,m)}-1} \log\pi_{\theta}(a^{(n,m)}_t|s^{(n,m)}_t),
\vspace{-3mm}$$ where
$A^{(n,m)}_{\text{GRPO}} = (\sum_{t=0}^{T-1}r^{(n,m)}_{t} - \text{mean}(\mathbf{R})) / \text{std}(\mathbf{R})$
with $\mathbf{R} = \{\sum_{t=0}^{T-1}r^{(n,m)}_{t}\}_{m\in[1,\dots,M]}$.
However, this approach has poor credit assignment for multi-turn
problems because all turns in the trajectory share the same advantage
estimation, and improving it typically requires tree-like sampling which
leads to combinatorial explosion. To bypass the expensive sampling from
each turn, we can learn a value function to estimate the return $G_t$,
known as *critic* (Sutton and Barto 2018), which in turn guides the
policy learning in the actor-critic architecture. We can compute
GAE (Schulman et al. 2015) for the advantage actor-critic (A2C)
objective:

$$\label{eq:a2c}
    \mathcal{J}_{\text{A2C}}(\theta) = \frac{1}{N} \sum_{n=1}^{N} \sum_{t=0}^{T^{(n)}-1} A_{\text{GAE}, t}^{(n)} \log\pi_{\theta}(a^{(n)}_t|s^{(n)}_t).
\vspace{-3mm}$$

To retain the benefits of fine-grained and stable advantage estimation
without the combinatorial explosion or learning an additional critic, we
instead use *Return Batch Normalization* (ReBN). For ReBN the
per-transition returns $G_i$ are normalized over the *whole batch of
transitions*: $$\label{eq:ours}
    \mathcal{J}_{\text{REINFORCE+ReBN}}(\theta) = \frac{1}{N} \sum_{n=1}^{N} \sum_{t=0}^{T^{(n)}-1} A_{\text{ReBN},t}^{(n)} \log\pi_{\theta}(a^{(n)}_t|s^{(n)}_t),
\vspace{-3mm}$$ where
$A^{(n)}_{\text{ReBN},t} = (G^{(n)}_t - \text{mean}(\mathbf{G})) / \text{std}(\mathbf{G})$,
with
$\mathbf{G} = \{G^{(n)}_t\}_{n\in[1,\dots,N],t\in[1,\dots,T^{(n)}-1]}$.
Each of these algorithms trains the agent by iterating between two main
phases: (A) data collection and (B) policy update. We present the RL
loop of in in due to space constraint.

# Empirical studies with GEM

In this section, we demonstrate how GEM can facilitate RL research on
agentic LLMs through a series of empirical studies. These include a
comprehensive apples-to-apples algorithm benchmarking across eight GEM
environments (); analyses of the effects of the discount factor $\gamma$
and tool integration (Sections
<a href="#sec:rl_language_game" data-reference-type="ref"
data-reference="sec:rl_language_game">4.2</a> and
<a href="#sec:rl_with_tools" data-reference-type="ref"
data-reference="sec:rl_with_tools">4.3</a>); an examination of
cross-task generalization (); and, finally, a demonstration of GEM’s
compatibility with five RL training frameworks along with their easily
accessible infrastructure benefits (). RL results in a vision-language
environment and analysis of a multi-agent environment can be found in .

## Benchmarking RL algorithms for LLMs

<span id="sec:algo_bench" label="sec:algo_bench"></span>

Benchmarking has been critical for the progress of RL, with OpenAI-Gym
providing standardized environments that enabled systematic evaluation
of algorithms (Raffin et al. 2021; Huang et al. 2022, 2024). Following
this paradigm, GEM offers a unified testbed for agentic LLMs, where
prior work often relied on bespoke tasks that complicate fair
comparison. We benchmark all baseline algorithms introduced in (GRPO,
PPO[^6], REINFORCE, ReBN) across eight GEM environments under a unified
experimental protocol. All algorithms are implemented using Oat (Z. Liu
et al. 2024) with hyperparameters detailed in . Results are evaluated by
mean episode return, sample efficiency, and stability.

We present all learning curves in . We first observe that in all three
single-turn environments (labeled with **rg**), GRPO performs reasonably
well, defending its effectiveness in single-step RL with verifiable
rewards. However, GRPO falls short when it comes to multi-turn
environments (`GuessTheNumber` and `Sudoku`), where dense per-turn
rewards are available and more fine-grained credit assignment is
necessary for efficient policy learning, due to a constant advantage
estimation across all steps. Such effects are the most profound when the
environment’s reward structure is inherently non-sparse (**qa** and
**math** is less so).

In contrast to GRPO, REINFORCE and PPO are natively suitable for
multi-turn RL. We find that vanilla REINFORCE is readily a strong
baseline in most environments, but it might suffer from suboptimal
convergence (e.g., two `Sudoku` environments). We hypothesize that this
might be because the raw return calculation of vanilla REINFORCE can be
sensitive to reward shaping, thus hindering exploration; we defer an
in-depth ablation study to . On the other hand, PPO is generally
performant, attaining the best episode return in the complex and
long-horizon `Sudoku` environment. This performance advantage can be
attributed to a well-learned critic, but it is also deemed difficult to
robustly learn an accurate critic (Van Hasselt et al. 2018; Kazemnejad
et al. 2025) (as evidenced by the poor performance of PPO in
`Minesweeper`), inviting future works to go in this direction.

<figure id="fig:algo_bench">
<embed src="imgs/algo_bench.pdf" />
<figcaption>Algorithm benchmarking using eight representative
environments from GEM. All agents are trained from
<code>Qwen3-{scale}-Base</code> models, with <code>scale</code>
specified in each plot. <strong>rg</strong> refers to single-turn
reasoning tasks from ReasoningGym; <strong>game</strong> consists of
long-horizon language games; <strong>qa</strong> and
<strong>math</strong> are tool-integrated multi-turn
environments.</figcaption>
</figure>

Finally, we investigate the proposed REINFORCE variant, which
incorporates a simple Return Batch Normalization (ReBN) technique.
Results in both show that ReBN consistently improves on vanilla
REINFORCE by a large margin, suggesting the empirical benefits of
adaptive normalization of policy gradient coefficients. Moreover, ReBN
outperforms or is comparable with PPO and GRPO in all evaluated
environments, rendering it the strongest baseline without expensive
computations, such as critic learning or extensive rollouts.

## Discount factor $\gamma$ matters

<figure id="fig:gamma_tool">
<embed src="imgs/gamma_tool.pdf" />
<figcaption><strong>(a)</strong> Average number of turns and episode
return when trained with different discount factors.
<strong>(b)</strong> Comparative experiment results on tool
availability.</figcaption>
</figure>

Next, we investigate the effect of the discount factor $\gamma$. A key
motivation for REINFORCE+ReBN over GRPO is its compatibility with
arbitrary discount factors. To investigate the effect of this we trained
the `Qwen3-1.7B-Base` model  (Yang et al. 2025) using REINFORCE+ReBN on
the `GuessTheNumber` environment. In this environment the agent must
guess a hidden number randomly selected between 1 and 50. At each turn
the agent may guess, and receives feedback as to whether the hidden
number is larger or smaller. The optimal strategy is therefore *binary
search*.

As shown in (a), as expected, smaller $\gamma$ values naturally
encourage solutions with fewer turns and drive convergence to the
optimal turn count ($\log_2(50)\approx5.6$)—achievable only through
binary search. Example interactions are included in . As discussed in ,
the natural efficiency incentive from $\gamma<1$ is not compatible with
GRPO. Instead, prior works using GRPO hyperparameter tune the
environment’s maximum number of turns to get efficient agent behavior
(Xue et al. 2025).

## Tool-integration in math and question-answering tasks

GEM is designed with modular support for external tools, enabling
seamless integration into a range of tasks. To empirically assess the
impact of tool use, we focus on two domains: Math and Question-Answering
(QA).

<div id="tab:math">

| Qwen3-4B-Base | Base (no tool) | Base (with tool) | Base + RL (no tool) | Base + RL (with tool) |
|:--------------|:--------------:|:----------------:|:-------------------:|:---------------------:|
| AIME24        |      10.0      |       6.7        |        16.7         |         30.0          |
| AMC           |      39.8      |       50.6       |        49.4         |         67.5          |
| MATH500       |      61.0      |       62.4       |        67.4         |         71.0          |
| MinervaMath   |      36.4      |       30.1       |        40.1         |         40.4          |
| OlympiadBench |      29.5      |       31.0       |        33.5         |         39.9          |
| Average       |      35.3      |       36.2       |        41.4         |       **49.8**        |

Math benchmark scores for four agents, evaluated with and without tool
access and RL training. Note: scores should be interpreted relative to
other values here due to sensitivity to the grader code (see ).

</div>

<span id="tab:math" label="tab:math"></span>

We first investigate the effect of GEM’s Python tool on Math tasks.
Starting from the base model `Qwen3-4B-Base`, we finetune on the
`math:Orz57K` environment, training two variants: one with Python tool
integration and one without. The base model and both finetuned models
are then evaluated across five distinct math environments.
Hyperparameter details are provided in , with the training curve shown
in (b), and Pass@1 accuracy reported in .

The math grader used for reward and evaluation is based on HuggingFace’s
`math_verify` library[^7]. We found that even minor differences in
grading logic across codebases yields substantial variation in reported
performance. Thus, all results should be interpreted
comparatively—within a consistent evaluation framework—rather than as
absolute values. This further highlights the need for unified
benchmarking, as provided by GEM.

Results in reveal a clear and consistent pattern: across all
environments, performance improves substantially after RL training
compared to the base model. Furthermore, the model with access to the
Python tool achieves higher final performance in every setting.

<span id="tab:qa" label="tab:qa"></span>

<div id="tab:qa">

|             |      |      |      |      |          |
|:------------|:----:|:----:|:----:|:----:|:--------:|
| Qwen3-4B    |      |      |      |      |          |
| (no tool)   |      |      |      |      |          |
| (no tool,   |      |      |      |      |          |
| single env) |      |      |      |      |          |
| (no tool,   |      |      |      |      |          |
| mixed env)  |      |      |      |      |          |
| (with tool, |      |      |      |      |          |
| single env) |      |      |      |      |          |
| (with tool, |      |      |      |      |          |
| mixed env)  |      |      |      |      |          |
| NQ†         | 6.1  | 15.4 | 15.8 | 35.0 |   37.3   |
| TriviaQA†   | 35.4 | 43.4 | 44.9 | 69.0 |   71.9   |
| PopQA†      | 11.3 | 19.0 | 19.9 | 47.1 |   48.1   |
| HotpotQA\*  | 11.1 | 21.1 | 22.1 | 43.2 |   45.5   |
| 2wiki\*     | 10.0 | 26.8 | 30.1 | 44.5 |   46.7   |
| Musique\*   | 2.9  | 4.7  | 5.5  | 17.6 |   19.9   |
| Bamboogle\* | 17.6 | 28.8 | 28.8 | 49.6 |   48.8   |
| Average     | 10.2 | 22.7 | 23.9 | 43.7 | **45.5** |

QA benchmark scores for the base agent and agents trained with different
RL configurations. † and \* denote single-hop and multi-hop datasets,
respectively.

</div>

We also perform a parallel analysis for QA tasks, this time integrating
the Search tool. We train on two environment compositions: `qa:HotpotQA`
alone, and a mixture of both `qa:HotpotQA` and `qa:NaturalQuestions`.
All other setting are the same as for the Math experiments (see above).
Evaluation spans seven diverse QA environments. Results, summarized in ,
mirror those from the math domain: RL finetuning markedly improves
performance, and models equipped with the Search tool achieve the
highest accuracy in every scenario.

The consistency of these findings across both domains (mathematics and
QA), tools (Python and Search), and multiple evaluation environments
underscores the flexibility and robustness of GEM’s approach to RL LLM
with tool integration.

## Studying generalization

<figure id="fig:game_generalization">
<embed src="imgs/game_generalization.pdf" />
<figcaption>Training on the <code>game:sudoku-v0-easy</code> environment
generalizes to ReasoningGym.</figcaption>
</figure>

GEM’s environments can be used for both training and evaluation. This
makes it ideal for investigating cross-environment generalization. For
instance, we demonstrate training on the `game:sudoku-v0-easy`
environment, while periodically evaluating on three different
environments, with some encouraging initial generalization results shown
in .

## Integration with training frameworks

Finally, we demonstrate that GEM—which takes care of the environment
side—can be easily integrated with five popular frameworks that handle
the training side. There has been a proliferation of frameworks focusing
on the training side of LLM RL. These often rely heavily on multiple
other libraries (such as vLLM for response generation (Kwon et al.
2023), and DeepSpeed for optimization (Rasley et al. 2020)). The diverse
range of features and design choices make it challenging for researchers
to select and adapt a suitable training framework to their specific
needs.

To address this GEM comes with complete, single-file training scripts
showing clean integration into five widely used LLM RL frameworks:
Oat (Z. Liu et al. 2024), Verl (Sheng et al. 2024), OpenRLHF (J. Hu et
al. 2024), ROLL (W. Wang et al. 2025), and RL2 (Tan et al. 2025). These
are validated in (a) where we show the training curve for each. Despite
minor differences due to underlying design choices of the frameworks
(e.g., different LLM generation engines) and RL stochasticity, all
curves exhibit similar trends, demonstrating that GEM is agnostic to
training frameworks and validating their implementation equivalence.
Furthermore, supporting a wide range of frameworks allows us to
effortlessly access their advanced features. For example, enabling the
asynchronous rollout in RL2 gives an immediate $2\times$ gain in
wall-clock efficiency ((b)).

<figure id="fig:framework_async">
<embed src="imgs/frameworks_async.pdf" />
<figcaption><strong>(a)</strong> Training curves on two environments
showing successful integration of GEM into five existing frameworks.
<strong>(b)</strong> Asynchronous rollout improves wall-clock efficiency
of training Sudoku-solving agents based on Qwen3-4B-Base.</figcaption>
</figure>

# Agent evaluation with GEM

<span id="sec:gem4eval" label="sec:gem4eval"></span>

In addition to RL training, GEM can serve as a **unified evaluation
interface** to test LLM agents’ performance. In this section, we present
two example use cases where we evaluate agents powered by strong LLMs
(GPT-5 (OpenAI 2025), Gemini-2.5-Pro (Gemini Team 2025) and
Claude-Sonnet-4 (Anthropic 2025b)) on two complex tasks: database
operation via model context protocol (MCP) (Anthropic 2025a) and
terminal interaction via docker containers, both of which have been
added to GEM following .

## General tool use via model context protocol

<div class="wrapfigure">

r0.55 <embed src="imgs/gem_eval.pdf" style="width:55.0%" />

</div>

Modern LLM agents often need to interact with external tools, such as
search engines, APIs, and code interpreters. To facilitate this, GEM is
designed to be compatible with the MCP, which is an open protocol that
provides a standardized way for LLMs to communicate with external tools
and data sources.

The MCP architecture consists of an MCP host (the LLM application), an
MCP client, and an MCP server (the external tool). By adopting this
protocol, GEM allows for "plug-and-play" tool usage, where any tool that
implements the MCP server interface can be used by an agent in a GEM
environment without custom integration. This significantly simplifies
the process of creating tool-augmented LLM agents and opens up a vast
ecosystem of potential tools.

Using a PostgreSQL MCP tool, we assess the agent’s tool-augmented
reasoning capabilities using 20 database tasks taken from MCPMark (T. M.
Team 2025). We report the average success rate and the average number of
turns required to complete the tasks in the left panel of [^8]. GPT-5
attains the best success rate with the fewest interactions, while
Gemini-2.5-Pro and Claude-Sonnet-4 have slightly lower and varied
performance.

## Terminal environment via Docker container

To support a wider range of tasks, especially those involving complex
software dependencies and interactions with the operating system, GEM
includes support for environments running inside docker containers. The
integrated terminal environment provides a sandboxed unix operating
system where agents can learn to perform tasks using shell commands.
This approach provides a high degree of isolation and reproducibility,
ensuring that the environment is consistent across different machines.

We assess the terminal mastery of LLM agents on $57$ tasks sampled from
Terminal-Bench (T. T.-B. Team 2025), without any scaffolding. The right
panel of reports the average success rate and the number of turns
required to complete the tasks. GPT-5 attains the highest success rate
with the most efficient interaction, followed by Claude-Sonnet-4 and
Gemini-2.5-Pro. The evaluation leverages the same interaction loop used
for RL training, highlighting GEM’s role as a unified framework for both
reinforcement learning and standardized evaluation.

# Conclusions

GEM aims to accelerate agentic LLM research by providing a decoupled and
clean library that is agnostic to training frameworks, a unified
agent-environment interface and a suite of standardized environments. In
this paper, we introduced the design choices of GEM, the current suite
of task domains and tools, features like vectorized environment
execution, a simple yet general multi-turn REINFORCE algorithm
implemented in five training frameworks, a comprehensive algorithm
benchmarking evaluation, and in-depth analysis on several algorithmic
details. We invite the community to enter the era of experience for LLM
agent learning, and join us in both using and continuing to develop the
GEM framework.

# Acknowledgment

This project was partially supported by the Singapore Ministry of
Education Academic Research Fund Tier 1 (Award Number: T1 251RES2514).

# Environment registration

GEM enables rapid development of new RL environments. In this section,
we illustrate two scenarios: (i) integrating additional datasets into an
existing task and (ii) defining a custom task, followed by the procedure
for registering these environments for use.

The following code snippet shows how to add a new dataset for math
environment, where the answer verification logic is predefined by GEM
and can be reused.

``` python
import gem
from gem.envs.registration import register

register(
    "math:GSM8K-Example",
    "gem.envs.math_env:MathEnv",
    dataset_name="axon-rl/GSM-8k", # HuggingFace or local dataset path
    question_key="problem",
    answer_key="answer",
)

env = gem.make("math:GSM8K-Example") # ready to use
```

Next, we demonstrate how to build a new environment from scratch by
defining the initial state distribution (in `.reset()`) and the
transition and reward functions (in `.step()`) as follows.

``` python
from gem.core import Env
from gem.envs.registration import register
from gem.utils.constants import TERMINAL_STATE
from gem.utils.parsing import extract_last_boxed_answer

class ReverseStringEnv(Env):
    def __init__(self, str_len: int = 5):
        super().__init__()
        self.str_len = str_len

    def _get_instructions(self):
        return (
            "You are tasked to reverse a given string.\n"
            "You may provide your response in any manner. Only the content wrapped inside \\boxed{} will be considered as your final answer.\n"
            f"Please reverse the string: {self.gt_str}.\n"
        )

    def reset(self, seed=None):
        super().reset(seed)
        characters = string.ascii_letters + string.digits  # A-Z, a-z, 0-9
        self.gt_str = "".join(random.choices(characters, k=self.str_len))
        return self._get_instructions(), {}

    def step(self, action):
        clean_action = extract_last_boxed_answer(action)
        if clean_action is None:
            reward = 0
        else:
            reward = float(clean_action[::-1] == self.gt_str)
        return TERMINAL_STATE, reward, True, True, {}


# Register your environment
register("custom:ReverseString", ReverseStringEnv)

env = gem.make("custom:ReverseString")
```

# Case studies of language games

This section illustrates example interactions of game-playing agents
trained on `GuessTheNumber` under different discount factors $\gamma$.
With a near-undiscounted setting ($\gamma=0.999$), the agent fails to
complete the game within the trial budget, as it lacks incentive to
obtain the terminal correctness reward as earlier as possible (see
example below).

<div class="tcolorbox">

<span style="color: red!20!violet">(**$\gamma=0.999$**, the agent
exhausts all $10$ trials)</span>

**State:** You are playing Guess The Number. You have to guess the
number between 1 and 50 (inclusive) within 50 turns. $\cdots$ As you
play, the history of your guesses will be appended below. Use the
information to complete the game before you run out of guesses.

Enter your first guess to start the game.

At turn 1, you guessed 14, and the target number is higher than 14.

At turn 2, you guessed 22, and the target number is higher than 22.

…

At turn 10, you guessed 39, which has been already guessed before.

Enter your next guess.

**Action:** Let’s break down the problem step by step:

1\. \*\*Initial Guess (Turn 1):\*\* The target number is higher than 14.
This means the target number is between 15 and 50 (inclusive).

2\. \*\*Initial Guess (Turn 2):\*\* The target number is higher than 22.
This means the target number is between 23 and 50 (inclusive).

…

From the final two steps, the target number is between 37 and 50
(inclusive). The final answer is $\backslash$boxed{38}.

</div>

In contrast, when using a more myopic return estimate ($\gamma=0.9$),
the agent is encouraged to minimize the number of turns, discovers the
binary search strategy, and converges to the optimal policy.

<div class="tcolorbox">

<span style="color: red!20!violet">(**$\gamma=0.9$**, the agent learns
to conduct binary search)</span>

**State:** You are playing Guess The Number. You have to guess the
number between 1 and 50 (inclusive) within 50 turns. $\cdots$ As you
play, the history of your guesses will be appended below. Use the
information to complete the game before you run out of guesses.

Enter your first guess to start the game.

At turn 1, you guessed 25, and the target number is lower than 25.

At turn 2, you guessed 12, and the target number is higher than 12.

At turn 3, you guessed 18, and the target number is higher than 18.

At turn 4, you guessed 21, and the target number is higher than 21.

At turn 5, you guessed 23, and the target number is lower than 23.

Enter your next guess.

**Action:** To determine the target number between 1 and 50, we can use
a **binary search strategy**, which is effective for narrowing down a
range when given hints like ’higher’ or ’lower.’

Step-by-Step Reasoning:

1\. \*\*Initial Guess (Turn 1):\*\* - The target number is lower than
25. - This means the target number is in the range \[1, 24\].

2\. \*\*Second Guess (Turn 2):\*\* - The target number is higher than
12. - This means the target number is in the range \[13, 24\].

3\. \*\*Third Guess (Turn 3):\*\* - The target number is higher than
18. - This means the target number is in the range \[19, 24\].

4\. \*\*Fourth Guess (Turn 4):\*\* - The target number is higher than
21. - This means the target number is in the range \[22, 24\].

5\. \*\*Fifth Guess (Turn 5):\*\* - The target number is lower than
23. - This means the target number is in the range \[22, 22\].

Final Guess:

The target number is in the range \[22, 22\], which means the target
number is 22. **Final Answer:** $\backslash$boxed{22}

</div>

# Algorithm

For completeness we include the full algorithm of ReBN in .

<div class="algorithm">

<div class="algorithmic">

Policy $\pi_\theta$, Environment
$\mathcal{M}=(\mathcal{S}, \mathcal{A}, P, R, \gamma, \rho)$, Batch size
$B$ Reset batch buffer $\mathcal{B} \leftarrow \emptyset$ **//
Multi-turn episode collection** Sample the initial state $s_0 \sim \rho$
$y_t \sim \pi_\theta(\cdot | s_t)$
$a_t \leftarrow \text{extract\_action}(y_t)$
$r_t \leftarrow R(s_t, a_t)$ $s_{t+1} \leftarrow P(s_t, a_t)$
$G_t \leftarrow \sum_{k=t}^{T-1} \gamma^{k-t} r_k$ Add $(s_t, y_t, G_t)$
to $\mathcal{B}$

**// Return Batch Normalization** **// Policy optimization** Update
$\theta$ using Monte Carlo policy gradient
$\sum_{i=1}^B \tilde{G_i}\nabla_\theta \log \pi_\theta(y_i|s_i)$

</div>

</div>

# Extended empirical studies with GEM

<figure id="fig:batch_return_norm">
<embed src="imgs/batch_return_norm.pdf" />
<figcaption>Learning curves of different reward shaping strategies.
(<strong>a-b</strong>) The average success rate of two environments.
(<strong>c-d</strong>) The corresponding average number of turns taken
to solve the tasks, equal to the number of tool calls minus
one.</figcaption>
</figure>

## Improving learning efficiency via return batch normalization (ReBN)

<span id="sec:batch_return_norm" label="sec:batch_return_norm"></span>
As briefly discussed in , while REINFORCE demonstrates strong
performance across most environments, its convergence can be suboptimal
in certain cases. To investigate this further, we present an in-depth
ablation study here. Following minimalist principles, we began with the
vanilla REINFORCE algorithm and a simple reward scheme: $r=1$ for
correct answers and $r=0$ otherwise. This approach has been shown
effective for single-turn RL training (Singh et al. 2023; Xiong et al.
2025). However, as shown in (c) (w.o ReBN), it failed to induce tool
usage in multi-turn settings, despite significant amount of initial
attempts.

We hypothesize that this failure arises from the absence of *negative
gradients* under 0/1 reward shaping, which are crucial for efficient
learning and exploration. To address this, we introduced negative
gradients in two ways: (i) assigning fixed negative rewards ($r=1$ for
correct and $r=-1$ for incorrect answers, denoted as Neg rew in ); and
(ii) applying Return Batch Normalization with 0/1 rewards, where Monte
Carlo returns in REINFORCE are normalized as described in (denoted as
ReBN in ). While both 0/1 and $\pm$<!-- -->1 reward schemes
theoretically induce the same optimal policy, they might exhibit
markedly different learning dynamics in practice.

Notably, ReBN demonstrates strong and consistent performance across
environments—not only in math and QA tasks () but also in all other
settings (). We also observe that models can be sensitive to fixed
reward shaping: for example, Neg rew fails to improve tool use in
`math:Orz57K`, yet leads to tool overuse in `qa:HotpotQA`, both of which
are suboptimal behaviors.

## RL on vision-language environments

In addition to text-only environments, we support visual elements as
part of the observation for the agent to understand and take actions. As
a demonstrative example, we build a visual-language environment based on
Geometry3k dataset[^9] for training reasoning agent to solve geometry
math problems with images input. We RL-tune Qwen2.5-VL-3B/7B-Instruct
(Bai et al. 2025) using Dr. GRPO (Z. Liu et al. 2025), and the learning
curves are shown in . An example reasoning trace is shown in .

<figure id="fig:vision_language_geometry3k">
<embed src="imgs/curves_vision_language_geometry3k-v2.pdf" />
<figcaption>Learning curves of vision-language agents. We RL-tune
Qwen2.5-VL-3B/7B-Instruct using Dr. GRPO on the
<code>math:Geometry3K</code> environment and track their training
rewards (left) and validation scores (right). </figcaption>
</figure>

<figure id="fig:geometry3k-validation0">
<embed src="imgs/geometry3k-validation0.pdf" />
<figcaption>An example problem and the response of a trained agent based
on Qwen2.5-VL-7B-Instruct. </figcaption>
</figure>

## Multi-agent environments

<figure id="fig:multi_agent_heatmap">
<embed src="imgs/multi_agent_heatmap.pdf" />
<figcaption>Multi-agent evaluation on TAU-bench retail. Stronger user
simulators (rows) consistently improve agent performance (columns)
across model strengths. </figcaption>
</figure>

Many scenarios considered within reinforcement learning, such as
adversarial games, social dilemmae and multi-party control systems
require multiple agents by definition (Albrecht, Christianos, and
Schäfer 2024). This introduces a whole host of different design
decisions which a general multi-agent API should be able to embrace. By
studying previous approaches to defining multi-agent APIs (Terry et al.
2021; Lanctot et al. 2019; Bou et al. 2023; S. Hu et al. 2023) and
condensing them into a simple abstraction scheme, GEM also provides a
high-level API for multi-agent LLM RL.

In doing so, GEM enables convenient development of agents that can
collaborate, compete, or simulate realistic interactions with other
agents/entities, adding comprehensiveness to its extensible platform.

**Design.** GEM’s multi-agent API is designed with the following
principles in mind:

- **Minimal Prescriptions.** The API only imposes constraints and
  methods when it is helpful to do so. This includes the `AgentSelector`
  class which allows for arbitrary definitions of turn order via
  overriding.

- **One Abstraction, One API.** There should be only one way to realize
  the environment abstraction regardless of the environment’s
  properties.

**API overview.** GEM provides a `MultiAgentEnv` base class that extends
the standard Gym API to support multiple agents. The `step()` and
`reset()` functions operate on dictionaries keyed by agent identifiers:

``` python
from gem.envs.multiagent import MultiAgentEnv

env = MyMultiAgentEnv()
observations, infos = env.reset()  # Dict[agent_id, observation]

while not done:
    actions = {agent_id: agent.act(obs) for agent_id, obs in observations.items()}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    done = all(terminations.values())
```

To implement a custom environment, users inherit from `MultiAgentEnv`
and implement `observe(agent)` and `_process_actions(actions)`. The
framework handles agent lifecycle management and cumulative rewards
tracking. Turn coordination is managed via `AgentSelector`, which
supports two modes: *sequential* (agents act one at a time in
round-robin order) and *parallel* (all agents act simultaneously). The
environment can then determine which agents are active at each step and
automatically advance turns, enabling flexible multi-agent interaction
patterns without manual bookkeeping.

**TAU-bench retail integration.** We demonstrate this API by integrating
the TAU-bench retail benchmark (Yao et al. 2024), which evaluates
conversational agents on customer service tasks. We formulate this as a
two-agent environment: an `assistant` agent using tools (order lookup,
product search) and a `user` agent simulating customer behavior via an
LLM. The user simulator is initialized with task instructions and
generates queries; the assistant must satisfy these requests before
episode termination.

**Impact of user model strength.** A key question in multi-agent RL is:
*how does simulated user agent capability affect trainable assistant
agent learning?* We vary both user and assistant models across three
levels: weak (Gemini-2.0-Flash-Lite), medium (GPT-4o-mini), and strong
(GPT-4o), yielding 9 configurations to study user-assistant model
interactions.

Evaluating across all 115 tasks from the TAU-bench retail test set (),
we find that stronger user agents consistently improve overall success
rates across all assistant agent model strengths. Notably, the strongest
assistant (GPT-4o) exhibits the largest absolute performance gains (20%
from weak to strong user), achieving 61.7% success with a strong user
simulator. Interestingly, a strong user paired with a weak assistant
(44.3%) outperforms a weak user paired with a strong assistant (41.7%),
demonstrating that improving the *user agent* is crucial for robust
conversational task completion. These results motivate us to develop
multi-agent RL to co-evolve user and assistant agents to achieve
scalable and autonomous learning.

# Related works

There is a significant body of work on tool-integrated language
models—including SkyRL-v0 (Cao et al. 2025), VerlTool (Jiang et al.
2025), ReCall and ReSearch (M. Chen et al. 2025), Search-R1 (Jin et al.
2025), ReTool (J. Feng et al. 2025), and SimpleTIR (Xue et al. 2025). A
common design pattern in these methods is to collect multi-turn
agent-environment interactions as single continuous sequences of tokens
of agent actions interleaved with environment outputs. Training then
simply involves masking the environment outputs from the loss
calculation.

However, this single-sequence approach presents two significant
limitations. First, the state observation is rigidly defined as the
complete history of actions and outputs. This restricts the ability to
use alternative state representations, such as pruning “thinking” tokens
or summarizing the history to avoid exceeding context lengths. Second,
this formulation inherently limits the reward structure to a single,
trajectory-level signal, preventing the use of finer-grained, per-step
rewards, and effectively fixing the discount factor at $\gamma=1$. In we
demonstrate that $\gamma<1$ is crucial for obtaining the optimal fastest
search behavior. By contrast, with trajectory-level rewards, the natural
speed incentive from $\gamma<1$ is lost, and hence other works, such as
SimpleTIR, must tune and enforce a strict turn-limit to get this
behavior.

To address this, our framework, GEM, is designed for maximum flexibility
by collecting trajectories as a sequence of individual transitions
(i.e., state, action, reward, next state) as in the full, unsimplified
RL formulation. This design choice enables arbitrary state observation
constructions (using observation wrappers), and also preserves
compatibility with per-turn rewards and arbitrary discount factors
$\gamma\leq1$. The verl-agent framework (L. Feng et al. 2025) also
adopts this transition-wise approach, which enables its implementation
of GiGPO (L. Feng et al. 2025), an RL method that utilizes turn-level
rewards. While GiGPO collapses to trajectory-level GRPO when
observations are unique, it is an example of a type of algorithm that is
now straightforward to implement with GEM’s infrastructure.

There are multiple popular frameworks that focus on the agent training
side (e.g., Oat (Z. Liu et al. 2024), Verl (Sheng et al. 2024), OpenRLHF
(J. Hu et al. 2024), ROLL (W. Wang et al. 2025), and RL2 (Tan et al.
2025)). Currently, many works that build on these, including verl-agent,
RAGEN (Z. Wang et al. 2025), Verlog (W. Chen et al. 2025), and many of
the works above, add environments by directly modifying the source code.
This results in tight coupling between training and environments, and
makes it difficult to maintain and reuse the environments for future
research. As a result, each codebase tends to support only a small,
ad-hoc collection of environments, making it hard to compare different
methods. Even environments with the same name are often inconsistent
between codebases. GEM addresses this by dealing with all the
environment infrastructure, including providing a diverse suite of
environments, and corresponding baselines. This makes it easy to keep
training and environments decoupled, with the aim of freeing researchers
from cumbersome environment development and setup, and thus enabling
quicker prototyping and evaluation of new ideas.

We also note that there are early works (Abdulhai et al. 2023; Tajwar et
al. 2025) incorporating text games as the evaluation toolkit or for
language model fine-tuning. However, they did not focus on a
standardized suite of RL training environments nor did they investigate
different RL algorithms.

# Experimental settings

All our experiments are performed on 8 $\times$ A100 GPUs and finished
in about one day. The detailed experimental configurations are shown in
.

<div id="tab:hp">

| Parameter                              | Value                                  |
|:---------------------------------------|:---------------------------------------|
| <span class="smallcaps">Actor</span>   |                                        |
| Maximum response length per turn       | $4096$ tokens                          |
| Sampling temperature, train            | 1.0                                    |
| Sampling temperature, evaluation       | 0.0                                    |
| (top P, top k)                         | (1.0, -1)                              |
| <span class="smallcaps">Learner</span> |                                        |
| Optimizer                              | AdamW                                  |
| Adam parameters ($\beta_1, \beta_2$)   | (0.9, 0.95)                            |
| Weight decay                           | 0.0                                    |
| Gradient norm clipping                 | 1.0                                    |
| Learning rate scheduler                | Constant                               |
| Learning rate                          | $1\times 10^{-6}$                      |
| Inner proximal update epoch            | 2                                      |
| KL loss coefficient                    | 0.0                                    |
| KL penalty coefficient                 | 0.0                                    |
| Policy clipping parameter              | 0.2                                    |
| Discount factor                        | 0.9 (**game**,**qa**); 1.0 (otherwise) |
| GAE $\lambda$                          | 0.95                                   |
| Steps                                  | 500                                    |

Hyperparameter configurations used in all experiments.

</div>

<span id="tab:hp" label="tab:hp"></span>

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-abdulhai2023lmrl" class="csl-entry">

Abdulhai, Marwa, Isadora White, Charlie Snell, Charles Sun, Joey Hong,
Yuexiang Zhai, Kelvin Xu, and Sergey Levine. 2023. “Lmrl Gym: Benchmarks
for Multi-Turn Reinforcement Learning with Language Models.” *arXiv
Preprint arXiv:2311.18232*.

</div>

<div id="ref-abe2003reinforcement" class="csl-entry">

Abe, Naoki, Alan W Biermann, and Philip M Long. 2003. “Reinforcement
Learning with Immediate Rewards and Linear Hypotheses.” *Algorithmica*
37 (4): 263–93.

</div>

<div id="ref-ahmadian2024back" class="csl-entry">

Ahmadian, Arash, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia
Kreutzer, Olivier Pietquin, Ahmet Üstün, and Sara Hooker. 2024. “Back to
Basics: Revisiting Reinforce Style Optimization for Learning from Human
Feedback in Llms.” *arXiv Preprint arXiv:2402.14740*.

</div>

<div id="ref-marl-book" class="csl-entry">

Albrecht, Stefano V., Filippos Christianos, and Lukas Schäfer. 2024.
*Multi-Agent Reinforcement Learning: Foundations and Modern Approaches*.
MIT Press. <https://www.marl-book.com>.

</div>

<div id="ref-andrychowicz2021matters" class="csl-entry">

Andrychowicz, Marcin, Anton Raichuk, Piotr Stańczyk, Manu Orsini, Sertan
Girgin, Raphaël Marinier, Leonard Hussenot, et al. 2021. “What Matters
for on-Policy Deep Actor-Critic Methods? A Large-Scale Study.” In
*International Conference on Learning Representations*.

</div>

<div id="ref-modelcontextprotocol" class="csl-entry">

Anthropic. 2025a. “Model Context Protocol.”
<https://github.com/modelcontextprotocol/modelcontextprotocol>; GitHub.

</div>

<div id="ref-anthropic2025claude" class="csl-entry">

———. 2025b. “System Card: Claude Opus 4 & Claude Sonnet 4.”
<https://www-cdn.anthropic.com/07b2a3f9902ee19fe39a36ca638e5ae987bc64dd.pdf>.

</div>

<div id="ref-bai2025qwen2" class="csl-entry">

Bai, Shuai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song,
Kai Dang, et al. 2025. “Qwen2. 5-Vl Technical Report.” *arXiv Preprint
arXiv:2502.13923*.

</div>

<div id="ref-torchrl" class="csl-entry">

Bou, Albert, Matteo Bettini, Sebastian Dittert, Vikash Kumar, Shagun
Sodhani, Xiaomeng Yang, Gianni De Fabritiis, and Vincent Moens. 2023.
“TorchRL: A Data-Driven Decision-Making Library for PyTorch.”
<https://arxiv.org/abs/2306.00577>.

</div>

<div id="ref-openaigym" class="csl-entry">

Brockman, Greg, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John
Schulman, Jie Tang, and Wojciech Zaremba. 2016. “OpenAI Gym.”

</div>

<div id="ref-cao2025skyrl" class="csl-entry">

Cao, Shiyi, Sumanth Hegde, Dacheng Li, Tyler Griggs, Shu Liu, Eric Tang,
Jiayi Pan, et al. 2025. “SkyRL-V0: Train Real-World Long-Horizon Agents
via Reinforcement Learning.”

</div>

<div id="ref-chen2025research" class="csl-entry">

Chen, Mingyang, Tianpeng Li, Haoze Sun, Yijie Zhou, Chenzheng Zhu,
Haofen Wang, Jeff Z. Pan, et al. 2025. “ReSearch: Learning to Reason
with Search for LLMs via Reinforcement Learning.” *arXiv Preprint
arXiv:2503.19470*.

</div>

<div id="ref-verlog" class="csl-entry">

Chen, Wentse, Jiayu Chen, Hao Zhu, and Jeff Schneider. 2025.
“Context-Lite Multi-Turn Reinforcement Learning for LLM Agents.” In
*ES-FoMo III: 3rd Workshop on Efficient Systems for Foundation Models*.
<https://openreview.net/forum?id=6CE5PLsZdW>.

</div>

<div id="ref-feng2025retool" class="csl-entry">

Feng, Jiazhan, Shijue Huang, Xingwei Qu, Ge Zhang, Yujia Qin, Baoquan
Zhong, Chengquan Jiang, Jinxin Chi, and Wanjun Zhong. 2025. “ReTool:
Reinforcement Learning for Strategic Tool Use in LLMs.” *arXiv Preprint
arXiv:2504.11536*.

</div>

<div id="ref-feng2025gigpo" class="csl-entry">

Feng, Lang, Zhenghai Xue, Tingcong Liu, and Bo An. 2025. “Group-in-Group
Policy Optimization for LLM Agent Training.” *arXiv Preprint
arXiv:2505.10978*.

</div>

<div id="ref-google2025gemini" class="csl-entry">

Gemini Team, Google. 2025. “Gemini 2.5: Pushing the Frontier with
Advanced Reasoning, Multimodality, Long Context, and Next Generation
Agentic Capabilities.” *arXiv Preprint arXiv:2507.06261*.

</div>

<div id="ref-guertler2025textarena" class="csl-entry">

Guertler, Leon, Bobby Cheng, Simon Yu, Bo Liu, Leshem Choshen, and
Cheston Tan. 2025. “TextArena.” *arXiv Preprint arXiv:2504.11442*.

</div>

<div id="ref-guo2025deepseek" class="csl-entry">

Guo, Daya, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin
Xu, Qihao Zhu, et al. 2025. “Deepseek-R1: Incentivizing Reasoning
Capability in Llms via Reinforcement Learning.” *arXiv Preprint
arXiv:2501.12948*.

</div>

<div id="ref-hu2024openrlhf" class="csl-entry">

Hu, Jian, Xibin Wu, Zilin Zhu, Xianyu, Weixun Wang, Dehao Zhang, and Yu
Cao. 2024. “OpenRLHF: An Easy-to-Use, Scalable and High-Performance RLHF
Framework.” *arXiv Preprint arXiv:2405.11143*.

</div>

<div id="ref-marllib" class="csl-entry">

Hu, Siyi, Yifan Zhong, Minquan Gao, Weixun Wang, Hao Dong, Xiaodan
Liang, Zhihui Li, Xiaojun Chang, and Yaodong Yang. 2023. “MARLlib: A
Scalable and Efficient Multi-Agent Reinforcement Learning Library.”
<https://arxiv.org/abs/2210.13708>.

</div>

<div id="ref-huang2022cleanrl" class="csl-entry">

Huang, Shengyi, Rousslan Fernand Julien Dossa, Chang Ye, Jeff Braga,
Dipam Chakraborty, Kinal Mehta, and João G. M. Araújo. 2022. “CleanRL:
High-Quality Single-File Implementations of Deep Reinforcement Learning
Algorithms.” *Journal of Machine Learning Research* 23 (274): 1–18.
<http://jmlr.org/papers/v23/21-1342.html>.

</div>

<div id="ref-huang2024open" class="csl-entry">

Huang, Shengyi, Quentin Gallouédec, Florian Felten, Antonin Raffin,
Rousslan Fernand Julien Dossa, Yanxiao Zhao, Ryan Sullivan, et al. 2024.
“Open Rl Benchmark: Comprehensive Tracked Experiments for Reinforcement
Learning.” *arXiv Preprint arXiv:2402.03046*.

</div>

<div id="ref-verltool" class="csl-entry">

Jiang, Dongfu, Zhuofeng Li, Yi Lu, Zhiheng Lvu, Ping Nie, Wenhu Chen,
Tianyu Pang, and Chao Du. 2025. “VerlTool.”
<https://github.com/TIGER-AI-Lab/verl-tool>.

</div>

<div id="ref-jin2025searchr1" class="csl-entry">

Jin, Bowen, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong
Wang, Hamed Zamani, and Jiawei Han. 2025. “Search-R1: Training Llms to
Reason and Leverage Search Engines with Reinforcement Learning.” *arXiv
Preprint arXiv:2503.09516*.

</div>

<div id="ref-kazemnejad2024vineppo" class="csl-entry">

Kazemnejad, Amirhossein, Milad Aghajohari, Eva Portelance, Alessandro
Sordoni, Siva Reddy, Aaron Courville, and Nicolas Le Roux. 2025.
“VinePPO: Refining Credit Assignment in RL Training of LLMs.” In
*International Conference on Machine Learning*.

</div>

<div id="ref-kwon2023efficient" class="csl-entry">

Kwon, Woosuk, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody
Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. “Efficient
Memory Management for Large Language Model Serving with Pagedattention.”
In *Proceedings of the 29th Symposium on Operating Systems Principles*,
611–26.

</div>

<div id="ref-lambert2024tulu" class="csl-entry">

Lambert, Nathan, Jacob Morrison, Valentina Pyatkin, Shengyi Huang,
Hamish Ivison, Faeze Brahman, Lester James V Miranda, et al. 2024. “Tulu
3: Pushing Frontiers in Open Language Model Post-Training.” *arXiv
Preprint arXiv:2411.15124*.

</div>

<div id="ref-openspiel" class="csl-entry">

Lanctot, Marc, Edward Lockhart, Jean-Baptiste Lespiau, Vinicius
Zambaldi, Satyaki Upadhyay, Julien Pérolat, Sriram Srinivasan, et al.
2019. “OpenSpiel: A Framework for Reinforcement Learning in Games.”
*CoRR* abs/1908.09453. <http://arxiv.org/abs/1908.09453>.

</div>

<div id="ref-liu2025spiral" class="csl-entry">

Liu, Bo, Leon Guertler, Simon Yu, Zichen Liu, Penghui Qi, Daniel
Balcells, Mickel Liu, et al. 2025. “SPIRAL: Self-Play on Zero-Sum Games
Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement
Learning.” *arXiv Preprint arXiv:2506.24119*.

</div>

<div id="ref-liu2025understanding" class="csl-entry">

Liu, Zichen, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du,
Wee Sun Lee, and Min Lin. 2025. “Understanding R1-Zero-Like Training: A
Critical Perspective.” In *Conference on Language Modeling (COLM)*.

</div>

<div id="ref-liu2024oat" class="csl-entry">

Liu, Zichen, Changyu Chen, Xinyi Wan, Chao Du, Wee Sun Lee, and Min Lin.
2024. “OAT: A Research-Friendly Framework for LLM Online Alignment.”
<https://github.com/sail-sg/oat>.

</div>

<div id="ref-openai2024openai" class="csl-entry">

OpenAI. 2024. “Openai O1 System Card.” *arXiv Preprint
arXiv:2412.16720*.

</div>

<div id="ref-openai2025gpt5" class="csl-entry">

———. 2025. “Gpt-5 System Card.”
<https://cdn.openai.com/gpt-5-system-card.pdf>.

</div>

<div id="ref-ouyang2022training" class="csl-entry">

Ouyang, Long, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright,
Pamela Mishkin, Chong Zhang, et al. 2022. “Training Language Models to
Follow Instructions with Human Feedback.” *Advances in Neural
Information Processing Systems* 35: 27730–44.

</div>

<div id="ref-stable-baselines3" class="csl-entry">

Raffin, Antonin, Ashley Hill, Adam Gleave, Anssi Kanervisto, Maximilian
Ernestus, and Noah Dormann. 2021. “Stable-Baselines3: Reliable
Reinforcement Learning Implementations.” *Journal of Machine Learning
Research* 22 (268): 1–8. <http://jmlr.org/papers/v22/20-1364.html>.

</div>

<div id="ref-rasley2020deepspeed" class="csl-entry">

Rasley, Jeff, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. 2020.
“Deepspeed: System Optimizations Enable Training Deep Learning Models
with over 100 Billion Parameters.” In *Proceedings of the 26th ACM
SIGKDD International Conference on Knowledge Discovery & Data Mining*,
3505–6.

</div>

<div id="ref-schulman2015high" class="csl-entry">

Schulman, John, Philipp Moritz, Sergey Levine, Michael Jordan, and
Pieter Abbeel. 2015. “High-Dimensional Continuous Control Using
Generalized Advantage Estimation.” *arXiv Preprint arXiv:1506.02438*.

</div>

<div id="ref-schulman2017proximal" class="csl-entry">

Schulman, John, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg
Klimov. 2017. “Proximal Policy Optimization Algorithms.” *arXiv Preprint
arXiv:1707.06347*.

</div>

<div id="ref-shao2024deepseekmath" class="csl-entry">

Shao, Zhihong, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi,
Haowei Zhang, et al. 2024. “Deepseekmath: Pushing the Limits of
Mathematical Reasoning in Open Language Models.” *arXiv Preprint
arXiv:2402.03300*.

</div>

<div id="ref-sheng2024hybridflow" class="csl-entry">

Sheng, Guangming, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru
Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. 2024. “HybridFlow: A
Flexible and Efficient RLHF Framework.” *arXiv Preprint arXiv:
2409.19256*.

</div>

<div id="ref-silver2025welcome" class="csl-entry">

Silver, David, and Richard S Sutton. 2025. “Welcome to the Era of
Experience.” *Google AI* 1.

</div>

<div id="ref-singh2023beyond" class="csl-entry">

Singh, Avi, John D Co-Reyes, Rishabh Agarwal, Ankesh Anand, Piyush
Patil, Xavier Garcia, Peter J Liu, et al. 2023. “Beyond Human Data:
Scaling Self-Training for Problem-Solving with Language Models.” *arXiv
Preprint arXiv:2312.06585*.

</div>

<div id="ref-stojanovski2025reasoning" class="csl-entry">

Stojanovski, Zafir, Oliver Stanley, Joe Sharratt, Richard Jones,
Abdulhakeem Adefioye, Jean Kaddour, and Andreas Köpf. 2025. “REASONING
GYM: Reasoning Environments for Reinforcement Learning with Verifiable
Rewards.” *arXiv Preprint arXiv:2505.24760*.

</div>

<div id="ref-sutton2018rlbook" class="csl-entry">

Sutton, Richard S., and Andrew G. Barto. 2018. *Reinforcement Learning:
An Introduction*. Second. The MIT Press.

</div>

<div id="ref-tajwar2025traininggenerallycuriousagent" class="csl-entry">

Tajwar, Fahim, Yiding Jiang, Abitha Thankaraj, Sumaita Sadia Rahman, J
Zico Kolter, Jeff Schneider, and Ruslan Salakhutdinov. 2025. “Training a
Generally Curious Agent.” <https://arxiv.org/abs/2502.17543>.

</div>

<div id="ref-Tan2025RL2" class="csl-entry">

Tan, Chenmien, Simon Yu, Lanbo Lin, Ze Zhang, Yuanwu Xu, Chenhao Jiang,
Tianyuan Yang, Sicong Xie, and Guannan Zhang. 2025. “RL2: Ray Less
Reinforcement Learning.” <https://github.com/ChenmienTan/RL2>.

</div>

<div id="ref-mcpmark_2025" class="csl-entry">

Team, The MCPMark. 2025. “MCPMark: Stress-Testing Comprehensive MCP
Use.” <https://github.com/eval-sys/mcpmark>.

</div>

<div id="ref-tbench_2025" class="csl-entry">

Team, The Terminal-Bench. 2025. “Terminal-Bench: A Benchmark for AI
Agents in Terminal Environments.”
<https://github.com/laude-institute/terminal-bench>.

</div>

<div id="ref-pettingzoo" class="csl-entry">

Terry, J. K., Benjamin Black, Nathaniel Grammel, Mario Jayakumar, Ananth
Hari, Ryan Sullivan, Luis Santos, et al. 2021. “PettingZoo: Gym for
Multi-Agent Reinforcement Learning.” <https://arxiv.org/abs/2009.14471>.

</div>

<div id="ref-van2018deep" class="csl-entry">

Van Hasselt, Hado, Yotam Doron, Florian Strub, Matteo Hessel, Nicolas
Sonnerat, and Joseph Modayil. 2018. “Deep Reinforcement Learning and the
Deadly Triad.” *arXiv Preprint arXiv:1812.02648*.

</div>

<div id="ref-wang2025reinforcement" class="csl-entry">

Wang, Weixun, Shaopan Xiong, Gengru Chen, Wei Gao, Sheng Guo, Yancheng
He, Ju Huang, et al. 2025. “Reinforcement Learning Optimization for
Large-Scale Learning: An Efficient and User-Friendly Scaling Library.”
*arXiv Preprint arXiv:2506.06122*.

</div>

<div id="ref-wang2025ragen" class="csl-entry">

Wang, Zihan, Kangrui Wang, Qineng Wang, Pingyue Zhang, Linjie Li,
Zhengyuan Yang, Xing Jin, et al. 2025. “RAGEN: Understanding
Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning.”
*arXiv Preprint arXiv:2504.20073*.

</div>

<div id="ref-williams1992simple" class="csl-entry">

Williams, Ronald J. 1992. “Simple Statistical Gradient-Following
Algorithms for Connectionist Reinforcement Learning.” *Machine Learning*
8: 229–56.

</div>

<div id="ref-xiong2025minimalist" class="csl-entry">

Xiong, Wei, Jiarui Yao, Yuhui Xu, Bo Pang, Lei Wang, Doyen Sahoo, Junnan
Li, et al. 2025. “A Minimalist Approach to Llm Reasoning: From Rejection
Sampling to Reinforce.” *arXiv Preprint arXiv:2504.11343*.

</div>

<div id="ref-xue2025simpletir" class="csl-entry">

Xue, Zhenghai, Longtao Zheng, Qian Liu, Yingru Li, Zejun Ma, and Bo An.
2025. “SimpleTIR: End-to-End Reinforcement Learning for Multi-Turn
Tool-Integrated Reasoning.” <https://simpletir.notion.site/report>.

</div>

<div id="ref-yang2025qwen3" class="csl-entry">

Yang, An, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng,
Bowen Yu, et al. 2025. “Qwen3 Technical Report.” *arXiv Preprint
arXiv:2505.09388*.

</div>

<div id="ref-yao2024tau" class="csl-entry">

Yao, Shunyu, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. 2024.
“$\tau$-Bench: A Benchmark for Tool-Agent-User Interaction in Real-World
Domains.” *arXiv Preprint arXiv:2406.12045*.

</div>

<div id="ref-ziegler2019fine" class="csl-entry">

Ziegler, Daniel M, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec
Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. 2019.
“Fine-Tuning Language Models from Human Preferences.” *arXiv Preprint
arXiv:1909.08593*.

</div>

</div>

[^1]: Code is available at: <https://github.com/axon-rl/gem>.

[^2]: $^\dag$Equal contribution with random order. $^*$Work done during
    their associate membership at Sea AI Lab.

[^3]: Ignoring token-level PPO clipping which has no effect if the
    updates are on-policy.

[^4]: Orthogonally, we can also utilize proximal updates (Schulman et
    al. 2017) to improve sample efficiency.

[^5]: This is not the original GRPO because we fixed the length bias as
    noted by (Z. Liu et al. 2025).

[^6]: PPO in this work generally refers to *turn-level PPO* instead of
    token-level PPO commonly seen in single-turn dialogue
    scenarios (Ouyang et al. 2022).

[^7]: [`github.com/huggingface/Math-Verify`](https://github.com/huggingface/Math-Verify)

[^8]: Our evaluation relies on the basic response generation API rather
    than agent frameworks (e.g., LangChain, OpenAI Agent SDK), which may
    lead to deviations from the original benchmark results.

[^9]: <https://huggingface.co/datasets/hiyouga/geometry3k>.
