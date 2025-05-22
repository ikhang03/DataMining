package util;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;

public class ConsoleOutputRedirector {
    private static PrintStream originalOut;
    private static PrintStream fileOut;

    /**
     * Redirects console output to the specified file
     * @param fileName The file to redirect output to
     * @throws IOException If file operations fail
     */
    public static void startRedirectToFile(String fileName) throws IOException {
        // Save the original System.out
        originalOut = System.out;

        // Create a new file output stream
        fileOut = new PrintStream(new FileOutputStream(fileName));

        // Create a custom PrintStream that writes to both console and file
        PrintStream multiOut = new MultiOutputStream(originalOut, fileOut);

        // Redirect System.out
        System.setOut(multiOut);
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