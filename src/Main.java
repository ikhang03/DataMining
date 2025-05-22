import model.*;
import preprocessing.dataImporter;
import util.ConsoleOutputRedirector;
import java.io.IOException;

public class Main {
    private static final String OUTPUT_FOLDER = "model_results";

    public static void main(String[] args) {
        try {
            // RandomForest
            System.out.println("=============RandomForest Classification=============");
            ConsoleOutputRedirector.startRedirectToFile("RandomForest_results.txt", OUTPUT_FOLDER);
            RandomForest();
            ConsoleOutputRedirector.stopRedirect();

            // RandomForestTuning
            System.out.println("=============RandomForestTuning Classification=============");
            ConsoleOutputRedirector.startRedirectToFile("RandomForestTuning_results.txt", OUTPUT_FOLDER);
            RandomForestTuning();
            ConsoleOutputRedirector.stopRedirect();

            // OneR
            System.out.println("=============OneR Classification=============");
            ConsoleOutputRedirector.startRedirectToFile("OneR_results.txt", OUTPUT_FOLDER);
            OneR();
            ConsoleOutputRedirector.stopRedirect();

            // IBk
            System.out.println("=============IBK Classification=============");
            ConsoleOutputRedirector.startRedirectToFile("IBk_results.txt", OUTPUT_FOLDER);
            IBk();
            ConsoleOutputRedirector.stopRedirect();

            // Naive Bayes
            System.out.println("=============Naive Bayes Classification=============");
            ConsoleOutputRedirector.startRedirectToFile("NaiveBayes_results.txt", OUTPUT_FOLDER);
            NB();
            ConsoleOutputRedirector.stopRedirect();

            //J48
            System.out.println("=============J48 Classification=============");
            ConsoleOutputRedirector.startRedirectToFile("J48_results.txt", OUTPUT_FOLDER);
            J48();
            ConsoleOutputRedirector.stopRedirect();

            System.out.println("=============J48 Tuning=============");
            ConsoleOutputRedirector.startRedirectToFile("J48Tuning_results.txt", OUTPUT_FOLDER);
            J48Tuning();
            ConsoleOutputRedirector.stopRedirect();

            // SVM
            System.out.println("=============SVM Classification=============");
            ConsoleOutputRedirector.startRedirectToFile("SVM_results.txt", OUTPUT_FOLDER);
            SVM();
            ConsoleOutputRedirector.stopRedirect();

            // Logistic Regression
            System.out.println("=============Logistic Regression Classification=============");
            ConsoleOutputRedirector.startRedirectToFile("LogisticRegression_results.txt", OUTPUT_FOLDER);
            LR();
            ConsoleOutputRedirector.stopRedirect();

            System.out.println("All model results have been saved to the '" + OUTPUT_FOLDER + "' folder.");

        } catch (IOException e) {
            System.err.println("Error redirecting output: " + e.getMessage());
            e.printStackTrace();
        }
    }

    public static void RandomForest() {
        (new RandomForestClassifier()).exec(dataImporter.trainSource, dataImporter.testSource);
    }

    public static void RandomForestTuning() {
        (new RandomForestTuning()).exec(dataImporter.trainSource, dataImporter.testSource);
    }

    public static void OneR() {
        (new OneRClassifier()).exec(dataImporter.trainSource, dataImporter.testSource);
    }

    public static void IBk() {
        (new IBkClassifier()).exec(dataImporter.trainSource, dataImporter.testSource);
    }

    public static void NB() {
        (new NaiveBayesClassifier()).exec(dataImporter.trainSource, dataImporter.testSource);
    }

    public static void J48() {
        (new J48Classifier()).exec(dataImporter.trainSource, dataImporter.testSource);
    }

    public static void J48Tuning() {
        (new J48Tuning()).exec();
    }

    public static void SVM() {
        (new SVMClassifier()).exec(dataImporter.trainSource, dataImporter.testSource);
    }

    public static void LR() {
        (new LogisticRegressionClassifier()).exec(dataImporter.trainSource, dataImporter.testSource);
    }
}