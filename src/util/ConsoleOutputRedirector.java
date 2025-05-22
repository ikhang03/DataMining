package util;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public class ConsoleOutputRedirector {
    private static PrintStream originalOut;
    private static PrintStream fileOut;
    private static final String DEFAULT_OUTPUT_FOLDER = "model_results";

    /**
     * Redirects console output to the specified file in the output folder
     * @param fileName The file to redirect output to
     * @throws IOException If file operations fail
     */
    public static void startRedirectToFile(String fileName) throws IOException {
        startRedirectToFile(fileName, DEFAULT_OUTPUT_FOLDER);
    }

    /**
     * Redirects console output to the specified file in the specified folder
     * @param fileName The file to redirect output to
     * @param folderName The folder to store the file in
     * @throws IOException If file operations fail
     */
    public static void startRedirectToFile(String fileName, String folderName) throws IOException {
        // Save the original System.out
        originalOut = System.out;

        File directory = new File(folderName);
        if (!directory.exists()) {
            if (!directory.mkdirs()) {
                throw new IOException("Failed to create directory: " + folderName);
            }
            System.out.println("Created directory: " + directory.getAbsolutePath());
        }

        // Add timestamp to filename to avoid overwrites
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        String fullPath = folderName + File.separator + fileName;

        fileOut = new PrintStream(new FileOutputStream(fullPath));

        fileOut.println("Model Results - Generated: " +
                LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")));
        fileOut.println("=====================================================");

        // Create a custom PrintStream that writes to both console and file
        PrintStream multiOut = new MultiOutputStream(originalOut, fileOut);

        // Redirect System.out
        System.setOut(multiOut);
        System.out.println("Results will be saved to: " + new File(fullPath).getAbsolutePath());
    }

    /**
     * Stops redirection and restores the original console output
     */
    public static void stopRedirect() {
        if (originalOut != null) {
            System.setOut(originalOut);
        }

        if (fileOut != null) {
            fileOut.close();
        }
    }

    /**
     * A custom PrintStream that can output to multiple streams
     */
    private static class MultiOutputStream extends PrintStream {
        private final PrintStream second;

        public MultiOutputStream(PrintStream first, PrintStream second) {
            super(first);
            this.second = second;
        }

        @Override
        public void write(int b) {
            super.write(b);
            second.write(b);
        }

        @Override
        public void write(byte[] buf, int off, int len) {
            super.write(buf, off, len);
            second.write(buf, off, len);
        }
    }
}