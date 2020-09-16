class ClassificationModelEvaluation:
    """
    Essa classe implementa funções que nos ajudarão a medir e armazenar o resultado das avaliações
    de modelos de classificação.
    """

    def __init__(self):
        pass

    def classification_metrics_in_dataframe(y_validation, y_pred, model=None):
        """
        Essa função retorna um dataframe com algumas métricas de avaliação de modelos de classificação.

        Args:
            y_validation (Series): Série com os valores da resposta usada para validação do modelo.
            y_pred (Series): Série com os valores da resposta prevista usada para validação do modelo.
            model (modelo instanciado, optional): Recebe o modelo instanciado do scikit learn. Se vazio, recebe None.

        Returns:
            DataFrame: retorna o DataFrame com algumas métricas de avaliação de modelos de classificação
        """
        import pandas as pd
        from sklearn.metrics import classification_report
        from datetime import datetime

        classification_report_dict = classification_report(y_validation, y_pred, output_dict=True)

        classification_metrics = classification_report_dict["macro avg"]
        classification_metrics["accuracy"] = classification_report_dict["accuracy"]

        model_name = str(model)[:str(model).find("(")]

        hash_evaluation_metric = hash((model_name, classification_metrics["f1-score"]))

        result = {hash_evaluation_metric: {"model": model_name, **classification_metrics,
                "params": str(model.get_params()), "date": datetime.today().date()}}

        result = pd.DataFrame.from_dict(result, orient="index")

        return result


    def save_model_metrics(result, path_processed=None):
        """
        Simplesmente salva em um arquivo csv as métricas dos modelos executados.
        Só salva se não há o modelo no arquivo csv.

        Args:
            result (DataFrame): DataFrame retornado da função classification_metrics_in_dataframe.
            path_processed (string, optional): String contendo o path de onde o arquivo deve ser salvo.
            Se usar o Cookiecutter de Data Science, a sugestão é que salve em "~/data/processed/model_evaluation.csv".

        Returns:
            DataFrame: DataFrame com o resd_csv do arquivo model_evaluation.csv salvo.
        """
        import pandas as pd

        try:
            result2 = pd.read_csv(path_processed+"model_evaluation.csv", index_col=0)
        except:
            result2 = pd.DataFrame(columns=['model', 'precision', 'recall', 'f1-score', 'support', 'accuracy','params', 'date'])
            result2.to_csv(path_processed+"model_evaluation.csv")

        if result.index in result2.index.tolist():
            date = result2.loc[int(result.index.values), "date"]
            print(f"O registro {int(result.index.values)} já foi salvo anteriormente no dia {date}.")
        else:
            print(f"O registro {result.index} foi salvo na base model_evaluation.csv.")
            result2 = result2.append(result)
            result2.to_csv(path_processed+"model_evaluation.csv")

        return result2



