import org.w3c.dom.Document;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import java.io.File;

public class batchReader {
    public static void main(String[] args) throws Exception{
        File files [] = new File("D:\\2011cut").listFiles((dir, name) -> {
            // TODO Auto-generated method stub
            return name.endsWith(".xml");
        });

        for (File file : files) {
            DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
            DocumentBuilder builder = factory.newDocumentBuilder();
            Document document = builder.parse(file);
            findFingerPrint.find(document);
        }
    }
}
