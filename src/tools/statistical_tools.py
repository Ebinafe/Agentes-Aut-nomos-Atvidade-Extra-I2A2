class StatisticalTools:
    """
    Ferramentas estatísticas
    """
    
    def describe(self, df):
        """
        Estatísticas descritivas
        """
        return df.describe()
    
    def correlation(self, df):
        """
        Matriz de correlação
        """
        return df.corr(numeric_only=True)
