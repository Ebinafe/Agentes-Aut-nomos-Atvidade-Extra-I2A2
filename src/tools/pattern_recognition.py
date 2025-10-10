class PatternRecognizer:
    """
    Reconhece padrões em dados categóricos
    """
    
    def find_patterns(self, df):
        """
        Retorna valores mais frequentes por coluna categórica
        """
        patterns = {}
        for col in df.select_dtypes(include='object').columns:
            patterns[col] = df[col].value_counts().head(5)
        return patterns
