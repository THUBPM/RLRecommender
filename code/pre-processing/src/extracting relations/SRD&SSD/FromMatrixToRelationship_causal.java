package processing;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import org.json.JSONObject;

public class FromMatrixToRelationship_causal {
	static String rootPath = "..\\data\\causal\\workflow\\";
	static String workflowFileName = "workflow.txt";
	static String outputRoot = "..\\data\\causal\\workflow\\";
	
	static String tripleAllPath = outputRoot + "all.txt";
	static String entityIndexPath = outputRoot + "entity2id.txt";
	static String relationIndexPath = outputRoot + "relation2id.txt";
	static String graphIndexPath = outputRoot + "graph2id.txt";
	
	static int count = 0;
	static Map<String, Integer> entityIndex = new HashMap<String, Integer>();
	static Map<String, Integer> relationIndex = new HashMap<String, Integer>();
	static Map<String, Integer> graphIndex = new HashMap<String, Integer>();
	static int entityCount = 0;
	static int relationCount = 0;
	static int graphCount = 0;
	
	public static void main(String[] args) throws IOException {
		GetData(rootPath);
		FileWriter relationshipTotalFile = new FileWriter(tripleAllPath);
		relationshipTotalFile.close();
		for(int i = 0; i < 5; i++) {
			relationshipTotalFile = new FileWriter(rootPath + "S" + (i % 5) + ".txt");
			relationshipTotalFile.close();
		}
		File file = new File(rootPath);
    	GetDirectory(file);
    	System.out.println("count==>" + count);
		WriteEntityIndex();
		WriteRelationIndex();
		WriteGraphIndex();
    }
	
	public static void GetDirectory(File file) {
		File flist[] = file.listFiles();
		if (flist == null) {
			return;
		}
		if (flist.length == 0) {
			file.delete();
			return;
		}
		for (File f : flist) {
			try {
				if (f.isDirectory()) {
//					System.out.println("Dir==>" + f.getAbsolutePath());
					GetDirectory(f);
				} else {
//					System.out.println("file==>" + f.getAbsolutePath());
					String fileName = f.getName();
					if(fileName.matches(".*_my_relationship.txt")) {
						// count
						count++;
						if(count % 10 == 0) {
							System.out.println("count==>" + count);
						}
						
						// generate data files
						WriteTripleSets(f.getAbsolutePath());
					
						WriteVersionSets(f.getAbsolutePath(), rootPath + "S" + (count % 5) + ".txt");
					}
			    }
			} catch (Exception e) {
				e.printStackTrace();
				System.out.println("File==>" + f.getAbsolutePath());
			}
		}
	}
	
    public static void WriteVersionSets(String relationshipFilePath, String triplePath) {
    	try {
			File file = new File(relationshipFilePath);
			BufferedReader reader = null;
			reader = new BufferedReader(new FileReader(file));
			String tempString = null;
			while ((tempString = reader.readLine()) != null) {
				String tempStrings[] = tempString.split("\t");
				
				if(!tempStrings[2].equals("/")) {					
					FileWriter relationshipTotalFile = new FileWriter(triplePath, true);
					relationshipTotalFile.write(tempString + "\n");
					relationshipTotalFile.close();
				}
			}
			reader.close();
    	} catch (IOException e) {
			e.printStackTrace();
		}
    }
	
    // all triple sets
    public static void WriteTripleSets(String relationshipFilePath) {
    	try {
			File file = new File(relationshipFilePath);
			BufferedReader reader = null;
			reader = new BufferedReader(new FileReader(file));
			String tempString = null;
			while ((tempString = reader.readLine()) != null) {
				String tempStrings[] = tempString.split("\t");
				
				if(!tempStrings[2].equals("/")) {
					if(!entityIndex.containsKey(tempStrings[0])) {
						entityIndex.put(tempStrings[0], entityCount);
						entityCount++;
					}
					
					if(!entityIndex.containsKey(tempStrings[1])) {
						entityIndex.put(tempStrings[1], entityCount);
						entityCount++;
					}
					
					if(!relationIndex.containsKey(tempStrings[2])) {
						relationIndex.put(tempStrings[2], relationCount);
						relationCount++;
					}
					
					if(!graphIndex.containsKey(tempStrings[3])) {
						graphIndex.put(tempStrings[3], graphCount);
						graphCount++;
					}
					
					FileWriter relationshipTotalFile = new FileWriter(tripleAllPath, true);
					relationshipTotalFile.write(tempString + "\n");
					relationshipTotalFile.close();
				}
			}
			reader.close();
    	} catch (IOException e) {
			e.printStackTrace();
		}
    }
	
    // entity index
    public static void WriteEntityIndex() {
    	try {
			FileWriter entityIndexFile = new FileWriter(entityIndexPath);
			
	    	Iterator<Map.Entry<String, Integer>> iterator = entityIndex.entrySet().iterator();
			while(iterator.hasNext()) {
				Map.Entry<String, Integer> entity = iterator.next();
				entityIndexFile.write(entity.getKey() + "\t" + entity.getValue() + "\n");
			}
			
			entityIndexFile.close();
    	} catch (IOException e) {
			e.printStackTrace();
		}
    }
    
    // relation index
    public static void WriteRelationIndex() {
    	try {
			FileWriter relationIndexFile = new FileWriter(relationIndexPath);
			
	    	Iterator<Map.Entry<String, Integer>> iterator = relationIndex.entrySet().iterator();
			while(iterator.hasNext()) {
				Map.Entry<String, Integer> relation = iterator.next();
				relationIndexFile.write(relation.getKey() + "\t" + relation.getValue() + "\n");
			}
			
			relationIndexFile.close();
    	} catch (IOException e) {
			e.printStackTrace();
		}
    }

    // graph index
    public static void WriteGraphIndex() {
    	try {
			FileWriter graphIndexFile = new FileWriter(graphIndexPath);
			
	    	Iterator<Map.Entry<String, Integer>> iterator = graphIndex.entrySet().iterator();
			while(iterator.hasNext()) {
				Map.Entry<String, Integer> graph = iterator.next();
				graphIndexFile.write(graph.getKey() + "\t" + graph.getValue() + "\n");
			}
			
			graphIndexFile.close();
    	} catch (IOException e) {
			e.printStackTrace();
		}
    }
    
	public static void GetData(String filePath) {
		File file = new File(filePath + workflowFileName);
		BufferedReader reader = null;
		try {
			reader = new BufferedReader(new FileReader(file));
			String tempString = null;
			tempString = reader.readLine();
			System.out.println(tempString);
			
			while ((tempString = reader.readLine()) != null) {
				String modelName = tempString;
				
				tempString = reader.readLine();
				int length = tempString.split(" ").length;
				int[][] edges = new int[length][length];
				int[] outs = new int[length];
				int[] ins = new int[length];
				
				int i = 0;
				while (!tempString.equals("#")) {
//					System.out.println(tempString);
					outs[i] = 0;
					String[] lineData = tempString.split(" ");
					for(int j = 0; j < length; j++) {
						if(lineData[j].equals("-"))
							edges[i][j] = 0;
						else {
							edges[i][j] = Integer.parseInt(lineData[j]);
							outs[i]++;
							ins[j]++;
						}
					}
					i++;
					tempString = reader.readLine();
				}
				
				// get relationship
				FileWriter fw = new FileWriter(filePath + modelName + "_my_relationship.txt");
				for(i = 0; i < length; i++) {
					for(int j = 0; j < length; j++) {
						if(edges[i][j] > 0 && i != j) {
							if(outs[i] > 2 && ins[j] > 2)
								fw.write(edges[i][i] + "\t" + edges[j][j] + "\t" + "directSometimesCausal/directSometimesInverseCausal" + "\t" + modelName + "\n");
							else if(outs[i] > 2)
								fw.write(edges[i][i] + "\t" + edges[j][j] + "\t" + "directAlwaysCausal/directSometimesInverseCausal" + "\t" + modelName + "\n");
							else if(ins[j] > 2)
								fw.write(edges[i][i] + "\t" + edges[j][j] + "\t" + "directSometimesCausal/directAlwaysInverseCausal" + "\t" + modelName + "\n");
							else
								fw.write(edges[i][i] + "\t" + edges[j][j] + "\t" + "directAlwaysCausal/directAlwaysInverseCausal" + "\t" + modelName + "\n");
						}
					}
				}
				fw.close();
			}
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (reader != null) {
				try {
					reader.close();
				} catch (IOException e1) {
					
				}
			}
		}
	}
}
