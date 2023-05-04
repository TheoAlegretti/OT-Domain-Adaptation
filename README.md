# OT-Domain-Adaptation
OT Domain Adaptation subject for ENSAE course

## Domain Adaptation: Definition and problem

**Domain Adaptation** is a major problem in Machine Learning, where the goal is to adapt a model trained on a source domain to a target domain, usually presenting differences in distribution. In most cases, models are trained on a specific dataset and may face poor performance when applied to a new dataset with different characteristics. Domain Adaptation aims to solve this problem by reducing the gap between these two domains and thus improving the model's performance on the target domain.

The main challenge of Domain Adaptation is to deal with the differences in distribution between the source and target domains. These differences may be due to variations in the examples' features, selection biases, or differences in observation conditions. When these differences are significant, models trained on the source domain may have poor performance on the target domain, making it difficult to use them in real-world applications.

## Optimal Transport: a solution for Domain Adaptation

In the paper "Optimal Transport for Domain Adaptation" by Aude Genevay, Gabriel Peyré, and Marco Cuturi, the authors propose using **Optimal Transport** theory to solve the Domain Adaptation problem. Optimal Transport is a mathematical method that determines the optimal way to transfer a probability distribution to another, minimizing a given transport cost.

The authors show that Optimal Transport can be used to align the source and target domain distributions by determining a transformation that minimizes the distance between these two domains. This transformation can then be applied to the source domain examples to adapt them to the target domain, thereby improving the model's performance.

## Objective of the Notebook

In this notebook, we will first recontextualize the Domain Adaptation problem and present a concrete example to illustrate its impact on Machine Learning models. Then, we will use the `ott-Jax` package to implement an Optimal Transport-based solution, building on the research paper mentioned earlier. We will show that Optimal Transport effectively solves the fitting issues induced by Domain Adaptation and thus improves the model's performance on the target domain. We will compare this package with the POT package to check the convergence of the OT method and the computational cost of each packages. 

## Mathematical Development

In their paper, the authors suggest that the domain adaptation problem can be seen as a graph matching problem. The use of optimal transport for this problem is based on the hypothesis that the domain drift is due to a transformation $T : \mathcal{X}^{source} \rightarrow \mathcal{X}^{target}$. This transformation can be interpreted as the push forward or transport map of $\mu$ to $\nu$, i.e., $T\#\mu = \nu$.

We can then derive a procedure to solve the adaptation problem:
1. Estimate $\mu$ and $\nu$ with the datasets
2. Estimate the transport map $T$ as defined above
3. Train a classifier on the transported source sample $T(\mathcal{X}^{source})$

As finding T from all possible transformations is intractable, we will introduce a cost function and solve the Monge or Kantorovitch problem. The Monge problem seeks to find a deterministic transport map that minimizes the cost function, while the Kantorovitch problem seeks to find a probabilistic transport plan.

The Kantorovitch formulation is more tractable and can be expressed as a linear optimization problem:

$$
\Pi^* = \arg\min_{\Pi \in \mathcal{U}(\mu, \nu)} \langle \Pi, C \rangle_F
$$

where $\Pi^*$ is the optimal transport plan, $\mathcal{U}(\mu, \nu)$ is the set of transport plans with marginals $\mu$ and $\nu$, $C$ is the cost matrix, and $\langle \cdot, \cdot \rangle$
