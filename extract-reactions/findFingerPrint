import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class findFingerPrint {
    static void find(Document document) throws Exception {

        ArrayList<String> reactionSmiles = new ArrayList<>();


        NodeList list = document.getChildNodes();
        Node node = list.item(0);
        NodeList reactionNode = node.getChildNodes();


        for(int i = 1; i < reactionNode.getLength(); i++) {

            Node nodeReaction = reactionNode.item(i);

            NodeList reactionSmileNode = nodeReaction.getChildNodes();
            Node node2 = reactionSmileNode.item(3);

            String reactionSmile = node2.getTextContent();
            reactionSmiles.add(reactionSmile);


            System.out.println(reactionSmile);
            i++;
        }

        export(reactionSmiles);
    }


    private static void export(ArrayList<String> list) throws IOException {
        File f = new File("d:/fingerprint.txt");

        BufferedWriter bw=new BufferedWriter(new FileWriter(f, true));
        for (String reaction : list) {
            bw.write(String.valueOf(reaction));
            bw.newLine();
        }
        bw.close();
    }

    private static Document getDocument() throws ParserConfigurationException,
            IOException, org.xml.sax.SAXException {
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        DocumentBuilder builder = factory.newDocumentBuilder();
        Document document = builder.parse(new File("src/largenets.xml"));
        Element e = document.getDocumentElement();
        return document;
    }

    public static void main(String[] args) throws Exception {
        Document doc = getDocument();
        find(doc);
    }
}
