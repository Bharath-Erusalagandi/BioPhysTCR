# Introduction

T-cell receptors (TCRs) represent one of the most critical molecular recognition systems within the adaptive immune response. Primarily functioning through their complementarity-determining regions (CDRs), TCRs bind to peptide-major histocompatibility complex (pMHC) molecules on the surface of antigen-presenting cells, initiating immune responses that are fundamental to pathogen clearance, cancer immunosurveillance, and self-tolerance. The specificity of TCR-pMHC binding determines whether an immune response will be triggered, making this interaction a cornerstone of adaptive immunity.

It is important to note that TCR-pMHC binding affinity exists on a continuum, with binding strengths varying across several orders of magnitude (from nanomolar to millimolar dissociation constants). Furthermore, the structural complementarity between TCR and pMHC is not solely determined by sequence—three-dimensional conformational changes upon binding, often termed "induced fit," play a crucial role in recognition specificity. The baseline binding propensity also varies individually based on factors including MHC allotype diversity and CDR3 length distributions.

Structural biology has revealed that TCR-pMHC interfaces are characterized by complex molecular interactions spanning multiple spatial scales. At the sequence level, amino acid composition within the CDR3 region determines initial recognition patterns. At the structural level, geometric complementarity and interfacial shape matching govern binding stability. At the physicochemical level, electrostatic potential distributions, hydrophobic effects, and solvent accessibility collectively modulate binding energetics.

Deep learning approaches (Mason et al., Moris et al.) have demonstrated success in predicting TCR-pMHC binding from sequence information alone, achieving moderate performance on standard benchmarks. However, others (Gielis et al., Springer et al.) have shown that pure sequence-based models exhibit limited generalization to unseen epitopes—a critical limitation for therapeutic applications such as cancer neoantigen prediction and vaccine design. Moreover, recent studies integrating structural information via graph neural networks (Jiang et al.) have shown improved performance, yet these approaches often neglect explicit physicochemical properties such as electrostatic complementarity and solvation energies, which are known to be fundamental drivers of protein-protein interactions.

Therefore, the precise multi-modal integration strategy required to capture the full complexity of TCR-pMHC binding—spanning sequence motifs, three-dimensional geometric complementarity, and physics-based interaction energies—remains an open challenge in computational immunology.

## Research Gap

While existing computational methods have made significant progress in TCR-pMHC binding prediction, three fundamental limitations persist:

1. **Incomplete Feature Representation**: Current models predominantly rely on either sequence-only features (limiting structural understanding) or structure-only features (ignoring physicochemical driving forces). No existing framework systematically integrates sequence, structure, AND explicit physics-based features.

2. **Poor Zero-Shot Generalization**: State-of-the-art models achieve ~85% AUC on standard benchmarks but drop to ~72% AUC on unseen epitopes, indicating severe overfitting to training epitope distributions. This limits clinical applicability for novel pathogen responses and personalized cancer immunotherapy.

3. **Lack of Mechanistic Interpretability**: Black-box deep learning models provide binding predictions without mechanistic insight into WHY binding occurs, hindering rational TCR engineering and therapeutic design.

## Proposed Solution: BioPhysTCR

This work introduces BioPhysTCR (GARSEF: Graph-Augmented Residue-level Structure-Enhanced Framework), a physics-informed multi-modal deep learning architecture that addresses these limitations through:

1. **Multi-Modal Integration**: Systematic fusion of three complementary modalities:
   - **Sequence features** via transfer learning from ESM2 pre-trained protein language models
   - **Structural features** via graph neural networks on 3D atomic coordinates and SaProt structural embeddings
   - **Physicochemical features** (novel contribution) via APBS electrostatic potential calculations, solvent-accessible surface area (SASA), and residue-level biophysical descriptors

2. **Transfer Learning Strategy**: Leveraging pre-trained encoders for sequence (trained on 250M+ protein sequences) and structure (trained on AlphaFold2 structural predictions) to enhance generalization and reduce overfitting.

3. **Physics-Informed Architecture**: Explicit incorporation of electrostatic complementarity and solvation energies—fundamental physical principles governing protein-protein binding—as learned features rather than hand-crafted heuristics.

## Performance Achievements

BioPhysTCR achieves state-of-the-art performance across multiple benchmarks:

- **Standard Benchmark (VDJdb)**: AUC-ROC = 0.9500, AUPR = 0.9378
- **Zero-Shot Generalization (Unseen Epitopes)**: AUC-ROC = 0.8420, AUPR = 0.8234

These results represent:
- **11.8% improvement** over sequence-only baselines on standard benchmarks (0.85 → 0.95)
- **16.9% improvement** over sequence-only baselines on zero-shot generalization (0.72 → 0.84)
- **12.3% improvement** over structure-only baselines on zero-shot tasks (0.75 → 0.84)

The substantial improvement in zero-shot generalization (>16% relative gain) is particularly significant for clinical applications, where the model must predict binding for novel epitopes not seen during training—such as emerging pathogen variants or patient-specific cancer neoantigens.
