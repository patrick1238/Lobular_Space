LOBULAR-SPACE
Multi-scale Spatial Characterisation of Immune Cell Topography in Reactive Lymphoid Tissue and Classical Hodgkin Lymphoma
________________________________________
1. Background and Rationale
The organisation of lymphoid tissue is inherently spatial and critically contributes to immune function. Within lymph nodes, effective immune responses depend on structured spatial arrangements of lymphocytes, antigen-presenting cells, and stromal components, enabling coordinated cell–cell interactions and signalling.
In reactive lymphadenitis, this organisation is largely preserved, resulting in a coherent microenvironment with recognisable compartmentalisation and regulated cellular distribution. While these features are well established in histopathology, they are predominantly described qualitatively and lack a systematic quantitative framework.
In contrast, classical Hodgkin lymphoma (cHL), particularly the mixed cellularity subtype, is characterised by rare but biologically dominant Hodgkin and Reed–Sternberg (HRS) cells (CD30⁺, often CD15⁺) embedded within a heterogeneous inflammatory infiltrate. These cells influence their surrounding microenvironment through cytokine signalling and immune modulation, potentially altering local tissue organisation.
A limitation in current research is the lack of a quantitative spatial reference for reactive lymphoid tissue. Without such a baseline, it is difficult to distinguish pathological alterations from normal variability. This project aims to contribute to this area by applying spatial analysis methods to characterise immune cell organisation in reactive and neoplastic tissue.
________________________________________
2. Aim and Objectives
The aim of this thesis is to develop and apply a Python-based spatial analysis pipeline to explore immune cell organisation and to relate computational findings to biologically interpretable tissue patterns.
The specific objectives are:
•	to characterise spatial organisation patterns in reactive lymphoid tissue 
•	to analyse global and local clustering behaviour 
•	to explore scale-dependent spatial structure 
•	to visualise spatial features within the tissue context (e.g. heatmaps, overlays) 
•	to interpret observed patterns in relation to immunological and pathological features 
•	to compare spatial characteristics between reactive lymphadenitis and cHL 
________________________________________



3. Methodological Approach
General principle
This project follows an exploratory and translational approach, combining spatial analysis with visualisation and biological interpretation.
Spatial features will be:
•	derived from single-cell coordinate data, 
•	visualised within the tissue context, and 
•	interpreted using basic immunological and histopathological concepts. 
The analysis focuses on a limited set of robust and interpretable spatial measures, ensuring feasibility within the timeframe of a Master’s thesis.
________________________________________
3.1 Global spatial organisation
Global spatial structure will be assessed using the Clark–Evans index, which quantifies clustering versus dispersion based on nearest-neighbour distances.
Results will be visualised and interpreted in relation to general tissue organisation, such as aggregated versus dispersed cell distributions.
________________________________________
3.2 Scale-dependent spatial structure
Ripley’s L-function will be used to explore clustering behaviour across different spatial scales. In addition, the F-function will provide a basic assessment of cell-free regions within the tissue.
Results will be visualised as clustering curves and spatial overlays, allowing qualitative interpretation of spatial patterns.
________________________________________
3.3 Spatial autocorrelation
Spatial autocorrelation will be analysed using:
•	Global Moran’s I, to assess overall similarity patterns 
•	Local Moran’s I (LISA), to identify local clusters and spatial outliers 
Results will be mapped onto the tissue to highlight potential regions of coordinated behaviour or local heterogeneity.
________________________________________
3.4 Comparative analysis
Spatial features will be compared between reactive Lymphnodes and classical Hodgkin lymphoma, Mixed type. Differences will be explored using non-parametric statistical tests (e.g. Mann–Whitney U test). The results will be interpreted cautiously, with the aim of identifying tendencies or differences in spatial organisation rather than establishing definitive biomarkers.
________________________________________
4. Data and Implementation
The analysis will be based on single-cell spatial coordinates (x/y/z) derived from Leica SP8 confocal microscopy (~20 µm), combined with features such as marker intensity and morphology. The pipeline will generate both quantitative outputs and visualisations, supporting interpretation of spatial patterns. The dataset comprises:
•	10 cases of reactive lymphadenitis 
•	10 cases of classical Hodgkin lymphoma (mixed cellularity subtype) 
stained against CD4 (T-helper cells), CD8 (cytotoxic T-cells), CD68 (M1 macrophages), CD163 (M2 macrophages), CD11c (Antigen presenting cells), CD30 (activated T/B cells/ HRS cells).

________________________________________
5. Expected Outcomes and Significance
This project is expected to provide:
•	an initial quantitative characterisation of spatial organisation in reactive lymphoid tissue 
•	an exploratory comparison with classical Hodgkin lymphoma 
•	a set of visual and computational tools for analysing spatial cell distributions 
By combining spatial analysis with biological interpretation, the study aims to contribute to a better understanding of tissue organisation and to provide a basis for future, more detailed investigations in spatial immuno-oncology.

