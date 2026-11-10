"""
Document Image Classification - DIC without Group-Wise training

Architecture : 
    - Backbone : LayoutLMv3-Base with exit point at layer [1,4,8,10,12]
    - Contrastive Heads : Non-Linear MLP (Linear -> GeLU -> Linear -> L2 Normalization )
    - Exit Classifiers : LayoutLMv3ClassificationHead (linear -> tanh -> dropout -> linear)

Two-Stage Protocol:
    - Stage 1 SupCon-ST pretraining:
        Optimizes Backbone Encoder and Projection Heads simultaneously using supervised Contrastive Loss
"""