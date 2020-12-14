import java.io.*;
import java.util.ArrayList;

public class helper {
    static ArrayList<String> fingerprints = new ArrayList<>();
    public static void main(String[] args) throws IOException {

        txtReader();
        exportReactions(fingerprints);
    }
    private static void txtReader(){
        File file = new File("D:\\compoundsWithFingerprint.txt");
        BufferedReader reader = null;
        String temp;
        String temp0;
        int i = 0;
        try {
            reader = new BufferedReader(new FileReader(file));
            while ((temp = reader.readLine()) != null) {
                System.out.print(i + "\n");
                temp0 = temp.replace("\"","");
                fingerprints.add(temp0);
                i++;
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
    }
    private static void exportReactions(ArrayList<String> list) throws IOException {
        File f = new File("d:/compoundsWithFingerprintMatrix.txt");
        BufferedWriter bw=new BufferedWriter(new FileWriter(f));
        for (String s : list) {
            bw.write(String.valueOf(s));
            bw.newLine();
        }
        bw.close();
    }
}
