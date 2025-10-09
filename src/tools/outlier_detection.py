<<<<<<< HEAD
class OutlierDetector:
    """
    Detecta outliers usando método IQR
    """
    
    def detect(self, df, column):
        """
        Retorna DataFrame com outliers
        """
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return df[(df[column] < lower) | (df[column] > upper)]
=======
class OutlierDetector:
    """
    Detecta outliers usando método IQR
    """
    
    def detect(self, df, column):
        """
        Retorna DataFrame com outliers
        """
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return df[(df[column] < lower) | (df[column] > upper)]
>>>>>>> d4ead596466e6316ed201c1b2862c7a17a1c2125
