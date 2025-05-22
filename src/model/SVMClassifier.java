package model;

import preprocessing.dataImporter;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.StringToNominal;
import weka.filters.unsupervised.attribute.NominalToBinary;

public class SVMClassifier implements Command {
    public static void main(String[] args) {
        Command cmd = new SVMClassifier();
        cmd.exec(dataImporter.trainSource, dataImporter.testSource);
    }

    @Override
    public void exec(DataSource trainSource, DataSource testSource) {
        try {
            Instances trainDataset = trainSource.getDataSet();
            Instances testDataset = testSource.getDataSet();

            if (trainDataset.classIndex() == -1) {
                trainDataset.setClassIndex(trainDataset.numAttributes() - 1);
            }

            if (testDataset.classIndex() == -1) {
                testDataset.setClassIndex(testDataset.numAttributes() - 1);
            }

            StringToNominal stringToNominal = new StringToNominal();
            stringToNominal.setInputFormat(trainDataset);
            stringToNominal.setOptions(new String[]{"-R", "2-3,4"}); // Adjust indices for protocol_type, service, flag
            trainDataset = Filter.useFilter(trainDataset, stringToNominal);
            testDataset = Filter.useFilter(testDataset, stringToNominal);

            NominalToBinary nominalToBinary = new NominalToBinary();
            nominalToBinary.setInputFormat(trainDataset);
            trainDataset = Filter.useFilter(trainDataset, nominalToBinary);
            testDataset = Filter.useFilter(testDataset, nominalToBinary);

            Normalize normalize = new Normalize();
            normalize.setInputFormat(trainDataset);
            trainDataset = Filter.useFilter(trainDataset, normalize);
            testDataset = Filter.useFilter(testDataset, normalize);

            int[] classCounts = new int[trainDataset.numClasses()];
            for (int i = 0; i < trainDataset.numInstances(); i++) {
                classCounts[(int) trainDataset.instance(i).classValue()]++;
            }

            double minorityRatio = Math.min(classCounts[0], classCounts[1]) / (double) Math.max(classCounts[0], classCounts[1]);

            if (minorityRatio < 0.5) {
                System.out.println("Applying SMOTE for class imbalance...");
                SMOTE smote = new SMOTE();
                smote.setInputFormat(trainDataset);
                trainDataset = Filter.useFilter(trainDataset, smote);
            }

            SMO svm = new SMO();
            svm.setC(1.0);
            svm.setBuildCalibrationModels(true);


            System.out.println("Building SVM classifier...");
            svm.buildClassifier(trainDataset);
            System.out.println("SVM parameters: " + String.join(" ", svm.getOptions()));

            System.out.println("Evaluating SVM classifier...");
            Evaluation eval = new Evaluation(trainDataset);
            eval.evaluateModel(svm, testDataset);

            System.out.println(eval.toSummaryString("\nResults\n======\n", false));
            System.out.println("Confusion Matrix:\n" + eval.toMatrixString());
            System.out.println("Correct % = " + eval.pctCorrect());
            System.out.println("Incorrect % = " + eval.pctIncorrect());
            System.out.println("AUC = " + eval.areaUnderROC(1));
            System.out.println("Kappa = " + eval.kappa());
            System.out.println("Precision = " + eval.precision(1));
            System.out.println("Recall = " + eval.recall(1));
            System.out.println("F-Measure = " + eval.fMeasure(1));
            System.out.println("Error Rate = " + eval.errorRate());
            System.out.println(eval.toClassDetailsString());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}