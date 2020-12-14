import java.io.*;
import java.util.ArrayList;

public class fingerPrintIndex {
    static int i = 0;
    static ArrayList<String> compounds = new ArrayList<>();
    static ArrayList<String> reactions = new ArrayList<>();
    public static void main(String[] args) throws IOException {

        txtReader();

        //Collections.sort(compounds);

        ArrayList<String> reactionsInt = reactionIndex(compounds, reactions);
        exportReactions(reactionsInt);

        ArrayList<String> compoundsInt = compoundsIndex(compounds);
        exportCompounds(compoundsInt);

        //System.out.println(reactionsInt);
        System.out.println("compounds: "+compounds.size());
        System.out.println("reactions: "+reactions.size());

    }
    private static ArrayList<String> reactionIndex (ArrayList<String> compounds, ArrayList<String> reactions){
        ArrayList<String> reactionsInt = new ArrayList<>();
        for(int i = 0; i<reactions.size();i++){
            String[] temp0 = reactions.get(i).split(";");
            String rea = String.valueOf(1+compounds.indexOf(temp0[0])) +" "+ String.valueOf(1+compounds.indexOf(temp0[1]));
            reactionsInt.add(rea);
        }
        return reactionsInt;
    }

    private static ArrayList<String> compoundsIndex (ArrayList<String> compounds){
        ArrayList<String> compoundsInt = new ArrayList<>();
        for(int i = 0; i<compounds.size();i++){
            String temp0 = compounds.get(i);
            String rea = 1+compounds.indexOf(temp0) +" "+ compounds.get(i);
            compoundsInt.add(rea);
        }
        return compoundsInt;
    }

    private static void txtReader(){
        File file = new File("D:/fingerLinks.txt");
        BufferedReader reader = null;
        String temp;

        try {
            reader = new BufferedReader(new FileReader(file));
            while ((temp = reader.readLine()) != null) {
                i++;
                String[] temp0 = temp.split(";");
                if (!temp.isEmpty()&&!temp.equals("\uFEFF0")&&temp0.length==2){
                    reactions.add(temp);

                    if(!compounds.contains(temp0[0])){
                        compounds.add(temp0[0]);
                    }
                    if(!compounds.contains(temp0[1])){
                        compounds.add(temp0[1]);
                    }
                }
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
    private static void exportCompounds(ArrayList<String> list) throws IOException {
        File f = new File("d:/fingerPrintsOfCompounds.txt");
        BufferedWriter bw=new BufferedWriter(new FileWriter(f));
        for (int i = 0; i<list.size();i++) {
            bw.write(String.valueOf(list.get(i)));
            bw.newLine();
        }
        bw.close();
    }
    private static void exportReactions(ArrayList<String> list) throws IOException {
        File f = new File("d:/fingerPrintIndexedLinks.txt");
        BufferedWriter bw=new BufferedWriter(new FileWriter(f));
        for (String s : list) {
            bw.write(String.valueOf(s));
            bw.newLine();
        }
        bw.close();
    }
}
