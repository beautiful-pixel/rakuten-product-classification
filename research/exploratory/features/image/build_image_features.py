from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA

from .color import ColorEncoder, MeanRGBTransformer, HistRGBTransformer
from .keypoint import CornerCounter, BoVWTransformer
from .shape import ParallelogramCounter
from .texture import HOGTransformer
from .intensity import MinMaxDiffTransformer
from .transforms import Flattener, Resizer, ProportionTransformer, ImageCleaner, CropTransformer



def build_image_features_pipeline():
    """
    Construit un pipeline de features visuelles combinant plusieurs familles
    de descripteurs : couleur, texture et forme.

    Returns:
        sklearn.pipeline.Pipeline: Pipeline de feature engineering image.
    """

    hog_pipeline = Pipeline([
        ('hog', HOGTransformer(orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2,2))),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=150))
    ])
    
    grayscale_pipeline = Pipeline([
        ('resizer', Resizer(dsize=(128, 128), grayscale=True)),
        ('union', FeatureUnion([
            ('corner_counter', CornerCounter(block_size=2, ksize=3)),
            ('parall_counter', ParallelogramCounter(
                min_perimeter=100,
                bins=[0,0.3,0.6,0.8,1.1]
            )),
            ('hog_pipeline', hog_pipeline),
            ('bovw', BoVWTransformer(n_clusters=100))
        ]))
    ])
    m, M = 15, 240
    max_diff_ranges = [[i*(M-m), (i+1)*(M-m)] for i in range(3)]
    hist_diff_pipeline = Pipeline([
        ('histrgb',  HistRGBTransformer(histSize=[M-m], ranges=[[m,M]]*3)),
        ('min_max_diff', MinMaxDiffTransformer(max_diff_ranges=max_diff_ranges)),
    ])

    colors_rate_pipeline = Pipeline([
        ('color_encoder', ColorEncoder()),
        ('flattener', Flattener()),
        ('proportion', ProportionTransformer(categories=range(3,12)))
    ])

    colors_pipeline = Pipeline([
        ('resizer', Resizer(dsize=(128, 128), grayscale=False)),
        ('union', FeatureUnion([
            ('histrgb', HistRGBTransformer(histSize=[16])),
            ('hist_diff_pipeline', hist_diff_pipeline),
            ('colors_rate_pipeline', colors_rate_pipeline),
        ]))
    ])

    lda_pipeline = Pipeline(steps=[('resizer', Resizer(dsize=(16,16))),
        ('flattener', Flattener()),
        ('scaler', StandardScaler()),
        ('lda', LDA(n_components=26))
    ])

    union = FeatureUnion(transformer_list=[
        ('grayscale_pipeline', grayscale_pipeline),
        ('colors_pipeline', colors_pipeline),
        ('lda_pipeline', lda_pipeline),
    ])

    pipeline = Pipeline([
        ('crop', CropTransformer()),
        ('union', union),
        ('scaler', StandardScaler()),
    ])

    return pipeline
