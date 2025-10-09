<<<<<<< HEAD
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
=======
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
>>>>>>> d4ead596466e6316ed201c1b2862c7a17a1c2125
