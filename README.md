# RGB-D Domain Adaptation for Object Recognition
A main drawback of deep models si the need of huge amount of data for a proper training. 
A partial solution is generating synthetic examples that does not require a costly labeling operation.  
However, this comes at a cost: *real images* belong to a slightly different domain with respect to *generated images*. 
This **domain shift** can be reduced with different techniques of *domain adaptation*.   

In this work we propose an improvement to the previous work of Robbiano et al. 
"Unsupervised domain adaptation through inter-modal rotation for rgb-d object recognition".


## Domain adaptation
Really popular in self driving cars, domain adaptation techniques allow to learn robust domain invariant features (real vs. fake, day vs. night).
Domain adaptation can be achieved roughtly in three ways:
1. Minimizing some distributional discrepancy measure, like Maximum Mean Discrepancy (MMD).
2. Through adversarial methods.
3. The third class leverages the solution of a self-supervised tasks in parallel with the main recognition task (this work).

## RGB-D images
This solution can be as well applied to the robotic vision field in which images come in both RGB and depth modalities, 
requiring the design of ad-hoc inter-modal self supervised tasks.
