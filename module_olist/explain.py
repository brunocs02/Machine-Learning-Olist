import shap as sh
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt

from module_olist.config import FIGURES_DIR, INTERIM_DATA_DIR, MODELS_DIR
from module_olist.modeling.predict import load_model
from module_olist.modeling.interpret import calculate_data_shap_values, create_explainer, prepare_data_for_shap

def main():

    model, model_name, threshold = load_model(MODELS_DIR/"trained_model.pkl", MODELS_DIR/"metadata.json")

    data = pd.read_csv(INTERIM_DATA_DIR/"dataset.csv")

    y = data["is_late"]

    X = data.drop(columns=["is_late"])
    X_sample = X.sample(n = 100, random_state=42)

    model, model_name, threshold = (
        load_model(
            model_path=(
                MODELS_DIR /
                "trained_model.pkl"
            ),
            metadata_path=(
                MODELS_DIR /
                "metadata.json"
            ),
        )
    )

    logger.info(f"{model_name} model loaded")

    X_transformed = prepare_data_for_shap(model, X_sample)

    explainer = create_explainer(model)
    shap_values = calculate_data_shap_values(explainer, X_transformed)

    logger.success("SHAP values calculated")

    sh.plots.bar(
        shap_values,
        max_display=10,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
         FIGURES_DIR /
        "shap_global_bar.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()
    
    sh.plots.beeswarm(
        shap_values,
        max_display=10,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR /
        "shap_beeswarm.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()
    
    sample_position = 0
    
    sh.plots.waterfall(
        shap_values[
            sample_position
        ],
        max_display=10,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR /
        "shap_waterfall.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    logger.success(
        "Gráficos SHAP gerados com sucesso."
    )

    logger.info(
        f"Gráficos salvos em: "
        f"{FIGURES_DIR}"
    )

if __name__ == "__main__":
    main()