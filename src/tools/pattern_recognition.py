<<<<<<< HEAD
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
=======
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
>>>>>>> d4ead596466e6316ed201c1b2862c7a17a1c2125
