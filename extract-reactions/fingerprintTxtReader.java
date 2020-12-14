import java.io.*;
import java.util.ArrayList;

public class fingerprintTxtReader {
    public static void main(String[] args) throws IOException {

        ArrayList<String> reactionsList = new ArrayList<>();
        ArrayList<String> compoundsList = new ArrayList<>();
        ArrayList<String> links = new ArrayList<>();
        ArrayList<String> links0 = new ArrayList<>();


        File file=new File("D:\\fingerprint.txt");
        BufferedReader reader=null;
        String temp;
        ArrayList<String> reactionLines = new ArrayList<>();

        try{
            reader=new BufferedReader(new FileReader(file));
            while((temp=reader.readLine())!=null){
                String t[] = temp.split(" |f");
                reactionLines.add(t[0]);
            }
        }
        catch(Exception e){
            e.printStackTrace();
        }
        finally{
            if(reader!=null){
                try{
                    reader.close();
                }
                catch(Exception e){
                    e.printStackTrace();
                }
            }
        }
        export(reactionLines);
        for(int i = 0; i<reactionLines.size();i++){
            String m = reactionLines.get(i);
            if(m.split(">").length>2&&m.split(">").length!=0) {
                String[] reaction = m.split(">");

                String[] reactantList = reaction[0].split("\\.");
                String[] productList = reaction[reaction.length-1].split("\\.");

                for (int r = 0; r < reactantList.length; r++) {
                    for (int k = 0; k < productList.length; k++) {
                        String linkEach = reactantList[r] + ";" + productList[k];
                        links.add(linkEach);
                    }
                }
            }
        }
        //for(int u = 0; u<links.size();u++)
        System.out.print(links.size());
        export(links);

    }
    private static void export(ArrayList<String> list) throws IOException {
        File f = new File("d:/fingerLinks.txt");

        BufferedWriter bw=new BufferedWriter(new FileWriter(f));
        for (String s : list) {
            bw.write(String.valueOf(s));
            bw.newLine();
        }
        bw.close();
    }
}
