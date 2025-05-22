package preprocessing;

import weka.core.converters.ConverterUtils.DataSource;

public class dataImporter {
    public static DataSource trainSource;
    public static DataSource testSource;
    public static DataSource validSource;

    static {
        try {
            trainSource = new DataSource("data/train_clean.arff");
            testSource = new DataSource("data/test_clean.arff");
            validSource = new DataSource("data/valid_clean.arff");
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
