import net.sourceforge.jFuzzyLogic.FIS;
import net.sourceforge.jFuzzyLogic.rule.FuzzyRuleSet;

public class FuzzyLogicApp {
    public static void main(String[] args) throws Exception {
        try {
            String fileName = args[0];
            int humidity = Integer.parseInt(args[1]);
            int temperature = Integer.parseInt(args[2]);
            int timeOfDay = Integer.parseInt(args[3]);
            int plantHeight = Integer.parseInt(args[4]);

            FIS fis = FIS.load(fileName, false);

            FuzzyRuleSet fuzzyRuleSet = fis.getFuzzyRuleSet();
            fuzzyRuleSet.chart();

            fuzzyRuleSet.setVariable("humidity", humidity);
            fuzzyRuleSet.setVariable("temperature", temperature);
            fuzzyRuleSet.setVariable("timeOfDay", timeOfDay);
            fuzzyRuleSet.setVariable("plantHeight", plantHeight);

            fuzzyRuleSet.evaluate();
//
            fuzzyRuleSet.getVariable("flowerModification").chartDefuzzifier(true);

//System.out.println(fuzzyRuleSet);

        } catch (ArrayIndexOutOfBoundsException ex) {
            System.out.println("Incorrect number of arguments. Example: java FuzzyLogicApp int<humidity> int<temperature> int<timeOfDay> int<plantHeight>");
        } catch (NumberFormatException ex) {
            System.out.println("Invalid parameter. Example: java FuzzyLogicApp int<humidity> int<temperature> int<timeOfDay> int<plantHeight>");
        }
    }
}
