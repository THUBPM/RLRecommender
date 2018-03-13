package processing;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.Map.Entry;

import org.json.JSONArray;
import org.json.JSONObject;

public class ExtractRelationship_concurrent {
	static String root = "..\\models_en";
//	static String root = "..\\models_en\\bpmn2.0\\2081504877\\BPMN2.0_Process\\2012-02-10_Old Process";
	static boolean coverOldFile = false;
	static String outputRoot = ".\\";
	
	static String tripleAllPath = outputRoot + "all.txt";
	static String entityIndexPath = outputRoot + "entity2id.txt";
	static String relationIndexPath = outputRoot + "relation2id.txt";
	static String graphIndexPath = outputRoot + "graph2id.txt";
	static String tripleTrainAllLastVersionPath = outputRoot + "train.txt";
	static String tripleTestAllLastVersionPath = outputRoot + "test.txt";
	static String tripleValidAllLastVersionPath = outputRoot + "valid.txt";
	
	static int count = 0;
	static int train_count = 0;
	static int test_count = 0;
	static int valid_count = 0;
	static Map<String, Integer> entityIndex = new HashMap<String, Integer>();
	static Map<String, Integer> relationIndex = new HashMap<String, Integer>();
	static Map<String, Integer> graphIndex = new HashMap<String, Integer>();
	static int entityCount = 0;
	static int relationCount = 0;
	static int graphCount = 0;
   
	public static void main(String[] args) throws IOException {
    	File file = new File(root);
		FileWriter relationshipTotalFile = new FileWriter(tripleTrainAllLastVersionPath);
		relationshipTotalFile.close();
		relationshipTotalFile = new FileWriter(tripleTestAllLastVersionPath);
		relationshipTotalFile.close();
		relationshipTotalFile = new FileWriter(tripleValidAllLastVersionPath);
		relationshipTotalFile.close();
		relationshipTotalFile = new FileWriter(tripleAllPath);
		relationshipTotalFile.close();
    	GetDirectory(file);
    	System.out.println("count==>" + count);
    	System.out.println("train_count==>" + train_count);
    	System.out.println("test_count==>" + test_count);
    	System.out.println("valid_count==>" + valid_count);
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
					if(fileName.matches("\\d*_rev\\d*_metadata.json")) {
						// get metadata
						JSONObject metadata = ReadJson(f.getAbsolutePath());
						int numberOfRevisions = metadata.getJSONObject("model").getInt("numberOfRevisions");
						int revisionNumber = metadata.getJSONObject("revision").getInt("revisionNumber");
						
						String isLastRevision = metadata.getJSONObject("revision").getString("isLastRevision");
						
						// count
						count++;
						if(count % 100 == 0) {
							System.out.println("count==>" + count);
						}
						
						// get petri net
						JSONObject petriNet = ReadJson(f.getParent() + "\\" + fileName.replaceAll("_metadata", "_my_petri"));
						
						// extract relationship
						File relationshipFile = new File(f.getParent() + "\\" + fileName.replaceAll("_metadata", "_my_relationship"));
						if(coverOldFile || !relationshipFile.exists()) {
							ExtractRelationshipFromPetriNet(petriNet, f.getParent() + "\\" + fileName.replaceAll("_metadata", "_my_petri"));
						}
						
						// generate data files
						WriteTripleSets(f.getParent() + "\\" + fileName.replaceAll("_metadata", "_my_relationship"));
					
						if(numberOfRevisions == 1) {
							WriteTrainAllLastVersionSets(f.getParent() + "\\" + fileName.replaceAll("_metadata", "_my_relationship"));
							train_count++;
						}
						else if(numberOfRevisions == 2) {
							if(revisionNumber == 1) {
								WriteTestAllLastVersionSets(f.getParent() + "\\" + fileName.replaceAll("_metadata", "_my_relationship"));
								test_count++;
							} else {
								WriteTrainAllLastVersionSets(f.getParent() + "\\" + fileName.replaceAll("_metadata", "_my_relationship"));
								train_count++;
							}
						}
						else {
						 	if(revisionNumber == 1) {
						 		WriteValidAllLastVersionSets(f.getParent() + "\\" + fileName.replaceAll("_metadata", "_my_relationship"));
						 		valid_count++;
						 	} else if(revisionNumber == 2) {
						 		WriteTestAllLastVersionSets(f.getParent() + "\\" + fileName.replaceAll("_metadata", "_my_relationship"));
						 		test_count++;
						 	} else {
						 		WriteTrainAllLastVersionSets(f.getParent() + "\\" + fileName.replaceAll("_metadata", "_my_relationship"));
						 		train_count++;
						 	}
						}
					}
			    }
			} catch (Exception e) {
				e.printStackTrace();
				System.out.println("File==>" + f.getAbsolutePath());
			}
		}
	}
	
	// train triple sets
	// amount all
	// split by last version
    public static void WriteTrainAllLastVersionSets(String relationshipFilePath) {
    	Set<String> set = new HashSet<String>();
    	try {
			File file = new File(relationshipFilePath);
			FileWriter relationshipTotalFile = new FileWriter(tripleTrainAllLastVersionPath, true);

			BufferedReader reader = null;
			reader = new BufferedReader(new FileReader(file));
			String tempString = null;
			while ((tempString = reader.readLine()) != null) {
				String tempStrings[] = tempString.split("\t");
				
				if(!tempStrings[2].equals("//")) {
					if(!set.contains(tempString)) {
						set.add(tempString);
						relationshipTotalFile.write(tempString + "\n");
					}
				}
			}
			reader.close();
			relationshipTotalFile.close();
    	} catch (IOException e) {
			e.printStackTrace();
		}
    }
    
	// test triple sets
	// amount all
	// split by last version
    public static void WriteTestAllLastVersionSets(String relationshipFilePath) {
    	Set<String> set = new HashSet<String>();
    	try {
			File file = new File(relationshipFilePath);
			FileWriter relationshipTotalFile = new FileWriter(tripleTestAllLastVersionPath, true);

			BufferedReader reader = null;
			reader = new BufferedReader(new FileReader(file));
			String tempString = null;
			while ((tempString = reader.readLine()) != null) {
				String tempStrings[] = tempString.split("\t");
				
				if(!tempStrings[2].equals("//")) {
					if(!set.contains(tempString)) {
						set.add(tempString);
						relationshipTotalFile.write(tempString + "\n");
					}
				}
			}
			reader.close();
			relationshipTotalFile.close();
    	} catch (IOException e) {
			e.printStackTrace();
		}
    }
    
	// valid triple sets
	// amount all
	// split by last version
    public static void WriteValidAllLastVersionSets(String relationshipFilePath) {
    	Set<String> set = new HashSet<String>();
    	try {
			File file = new File(relationshipFilePath);
			FileWriter relationshipTotalFile = new FileWriter(tripleValidAllLastVersionPath, true);

			BufferedReader reader = null;
			reader = new BufferedReader(new FileReader(file));
			String tempString = null;
			while ((tempString = reader.readLine()) != null) {
				String tempStrings[] = tempString.split("\t");
				
				if(!tempStrings[2].equals("//")) {
					if(!set.contains(tempString)) {
						set.add(tempString);
						relationshipTotalFile.write(tempString + "\n");
					}
				}
			}
			reader.close();
			relationshipTotalFile.close();
    	} catch (IOException e) {
			e.printStackTrace();
		}
    }
	
    // all triple sets
    public static void WriteTripleSets(String relationshipFilePath) {
    	Set<String> set = new HashSet<String>();
    	try {
			File file = new File(relationshipFilePath);
			FileWriter relationshipTotalFile = new FileWriter(tripleAllPath, true);

			BufferedReader reader = null;
			reader = new BufferedReader(new FileReader(file));
			String tempString = null;
			while ((tempString = reader.readLine()) != null) {
				String tempStrings[] = tempString.split("\t");
				
				if(!tempStrings[2].equals("//")) {
					if(!set.contains(tempString)) {
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
						
						set.add(tempString);
						relationshipTotalFile.write(tempString + "\n");
					}
				}
			}
			reader.close();
			relationshipTotalFile.close();
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
	
	// extract relationship file
	public static void ExtractRelationshipFromPetriNet(JSONObject petriNet, String filename) {
    	class ModelObject {
    		public String name = null;
    		public String id = null;
    		public String type = null;
    		public String originalType = null;
    		public String loop = null;
    		public Set<String> contains = null;
    		public Set<String> outgoing = null;
    		public Set<String> ingoing = null;
    		
    		public ModelObject(JSONObject oneObject) {
    			super();
    			this.name = oneObject.getString("name");
    			this.id = oneObject.getString("id");
    			this.type = oneObject.getString("type");
    			this.loop = oneObject.getString("loop");
    			this.originalType = oneObject.getString("originalType");
    			
    			this.contains = new HashSet<String>();
    			JSONArray contains = oneObject.getJSONArray("contains");
    			for(int i = 0; i < contains.length(); i++) {
    				this.contains.add(contains.getString(i));
    			}
    			
    			this.outgoing = new HashSet<String>();
    			JSONArray outgoing = oneObject.getJSONArray("outgoing");
    			for(int i = 0; i < outgoing.length(); i++) {
    				this.outgoing.add(outgoing.getString(i));
    			}
    			
    			this.ingoing = new HashSet<String>();
    			JSONArray ingoing = oneObject.getJSONArray("ingoing");
    			for(int i = 0; i < ingoing.length(); i++) {
    				this.ingoing.add(ingoing.getString(i));
    			}
    		}
    	}
    	
    	// load json
    	Map<String, Integer> modelObjectsIndex = new HashMap<String, Integer>();
    	Map<Integer, String> modelObjectsInverseIndex = new HashMap<Integer, String>();
    	Map<String, ModelObject> modelObjects = new HashMap<String, ModelObject>();
    	int modelObjectsIndexCount = 0;
    	JSONArray list = petriNet.getJSONArray("list");
    	for(int i = 0; i < list.length(); i++) {
    		JSONObject oneObject = list.getJSONObject(i);
			modelObjects.put(oneObject.getString("id"), new ModelObject(oneObject));
			if(!oneObject.getString("type").equals("Place")) {
				modelObjectsIndex.put(oneObject.getString("id"), modelObjectsIndexCount);
				modelObjectsInverseIndex.put(modelObjectsIndexCount, oneObject.getString("id"));
				modelObjectsIndexCount++;
			}
    	}

    	// direct relationship
    	String directCausal[][] = new String[modelObjectsIndexCount][modelObjectsIndexCount];
    	String directInverseCausal[][] = new String[modelObjectsIndexCount][modelObjectsIndexCount];
    	String directConcurrent[][] = new String[modelObjectsIndexCount][modelObjectsIndexCount];
    	for(int i = 0; i < modelObjectsIndexCount; i++)
    		for(int j = 0; j < modelObjectsIndexCount; j++) {
    			directCausal[i][j] = "";
    			directInverseCausal[i][j] = "";
    			directConcurrent[i][j] = "";
    		}
    	
    	Iterator<Map.Entry<String, ModelObject>> iterator = modelObjects.entrySet().iterator();
		while(iterator.hasNext()) {
			Map.Entry<String, ModelObject> object = iterator.next();
    		ModelObject oneObject = object.getValue();
    		
    		if(oneObject.type.equals("Transition") || oneObject.type.equals("VerticalEmptyTransition")) {
	    		Iterator<String> iterator2 = oneObject.outgoing.iterator();
				while(iterator2.hasNext()) {
					String outgoing = iterator2.next();
					ModelObject nextObject = modelObjects.get(outgoing);
	    			if(nextObject.type.equals("Place")) {
    					Iterator<String> iterator3 = nextObject.outgoing.iterator();
    					while(iterator3.hasNext()) {
    						String nextOutgoing = iterator3.next();
    						ModelObject nextNextObject = modelObjects.get(nextOutgoing);
    		    			if(nextNextObject.type.equals("Transition") || nextNextObject.type.equals("VerticalEmptyTransition")) {
//    		    				if(nextObject.outgoing.size() > 1 && nextObject.ingoing.size() > 1) {
//    		    					directCausal[modelObjectsIndex.get(oneObject.id)][modelObjectsIndex.get(nextNextObject.id)] = "directSometimesCausal";
//    		    					directInverseCausal[modelObjectsIndex.get(oneObject.id)][modelObjectsIndex.get(nextNextObject.id)] = "directSometimesInverseCausal";
//    		    				} else if(nextObject.ingoing.size() > 1) {
//    		    					if(directCausal[modelObjectsIndex.get(oneObject.id)][modelObjectsIndex.get(nextNextObject.id)].equals("directSometimesCausal"))
//    		    						directCausal[modelObjectsIndex.get(oneObject.id)][modelObjectsIndex.get(nextNextObject.id)] = "directSometimesCausal";
//    		    					else
//    		    						directCausal[modelObjectsIndex.get(oneObject.id)][modelObjectsIndex.get(nextNextObject.id)] = "directAlwaysCausal";
//    		    					directInverseCausal[modelObjectsIndex.get(oneObject.id)][modelObjectsIndex.get(nextNextObject.id)] = "directSometimesInverseCausal";
//    		    				} else if(nextObject.outgoing.size() > 1) {
//    		    					directCausal[modelObjectsIndex.get(oneObject.id)][modelObjectsIndex.get(nextNextObject.id)] = "directSometimesCausal";
//    		    					if(directInverseCausal[modelObjectsIndex.get(oneObject.id)][modelObjectsIndex.get(nextNextObject.id)].equals("directSometimesInverseCausal"))
//    		    						directInverseCausal[modelObjectsIndex.get(oneObject.id)][modelObjectsIndex.get(nextNextObject.id)] = "directSometimesInverseCausal";
//    		    					else
//    		    						directInverseCausal[modelObjectsIndex.get(oneObject.id)][modelObjectsIndex.get(nextNextObject.id)] = "directAlwaysInverseCausal";
//    		    				} else {
//    		    					if(directCausal[modelObjectsIndex.get(oneObject.id)][modelObjectsIndex.get(nextNextObject.id)].equals("directSometimesCausal"))
//    		    						directCausal[modelObjectsIndex.get(oneObject.id)][modelObjectsIndex.get(nextNextObject.id)] = "directSometimesCausal";
//    		    					else
//    		    						directCausal[modelObjectsIndex.get(oneObject.id)][modelObjectsIndex.get(nextNextObject.id)] = "directAlwaysCausal";
//    		    					if(directInverseCausal[modelObjectsIndex.get(oneObject.id)][modelObjectsIndex.get(nextNextObject.id)].equals("directSometimesInverseCausal"))
//    		    						directInverseCausal[modelObjectsIndex.get(oneObject.id)][modelObjectsIndex.get(nextNextObject.id)] = "directSometimesInverseCausal";
//    		    					else
//    		    						directInverseCausal[modelObjectsIndex.get(oneObject.id)][modelObjectsIndex.get(nextNextObject.id)] = "directAlwaysInverseCausal";
//    		    				}
    		    				if(nextObject.outgoing.size() > 1 && nextObject.ingoing.size() > 1) {
    		    					if(directCausal[modelObjectsIndex.get(oneObject.id)][modelObjectsIndex.get(nextNextObject.id)].equals("directAlwaysCausal"))
    		    						directCausal[modelObjectsIndex.get(oneObject.id)][modelObjectsIndex.get(nextNextObject.id)] = "directAlwaysCausal";
    		    					else
    		    						directCausal[modelObjectsIndex.get(oneObject.id)][modelObjectsIndex.get(nextNextObject.id)] = "directSometimesCausal";
    		    					if(directInverseCausal[modelObjectsIndex.get(oneObject.id)][modelObjectsIndex.get(nextNextObject.id)].equals("directAlwaysInverseCausal"))
    		    						directInverseCausal[modelObjectsIndex.get(oneObject.id)][modelObjectsIndex.get(nextNextObject.id)] = "directAlwaysInverseCausal";
    		    					else
    		    						directInverseCausal[modelObjectsIndex.get(oneObject.id)][modelObjectsIndex.get(nextNextObject.id)] = "directSometimesInverseCausal";
    		    				} else if(nextObject.ingoing.size() > 1) {
   		    						directCausal[modelObjectsIndex.get(oneObject.id)][modelObjectsIndex.get(nextNextObject.id)] = "directAlwaysCausal";
    		    					if(directInverseCausal[modelObjectsIndex.get(oneObject.id)][modelObjectsIndex.get(nextNextObject.id)].equals("directAlwaysInverseCausal"))
    		    						directInverseCausal[modelObjectsIndex.get(oneObject.id)][modelObjectsIndex.get(nextNextObject.id)] = "directAlwaysInverseCausal";
    		    					else
    		    						directInverseCausal[modelObjectsIndex.get(oneObject.id)][modelObjectsIndex.get(nextNextObject.id)] = "directSometimesInverseCausal";
      		    				} else if(nextObject.outgoing.size() > 1) {
      		    					if(directCausal[modelObjectsIndex.get(oneObject.id)][modelObjectsIndex.get(nextNextObject.id)].equals("directAlwaysCausal"))
    		    						directCausal[modelObjectsIndex.get(oneObject.id)][modelObjectsIndex.get(nextNextObject.id)] = "directAlwaysCausal";
    		    					else
    		    						directCausal[modelObjectsIndex.get(oneObject.id)][modelObjectsIndex.get(nextNextObject.id)] = "directSometimesCausal";
    		    					directInverseCausal[modelObjectsIndex.get(oneObject.id)][modelObjectsIndex.get(nextNextObject.id)] = "directAlwaysInverseCausal";
    		    				} else {
   		    						directCausal[modelObjectsIndex.get(oneObject.id)][modelObjectsIndex.get(nextNextObject.id)] = "directAlwaysCausal";
   		    						directInverseCausal[modelObjectsIndex.get(oneObject.id)][modelObjectsIndex.get(nextNextObject.id)] = "directAlwaysInverseCausal";
    		    				}
    		    			} else
    		    				System.out.println("Error: Place after Place " + nextObject.id + " " + nextNextObject.id + " " + filename);
    		    		}
	    			} else
	    				System.out.println("Error: Transition after Transition " + oneObject.id + " " + nextObject.id + " " + filename);
	    		}
    		}
    		
    		if(oneObject.type.equals("Transition") || oneObject.type.equals("VerticalEmptyTransition")) {
	    		Iterator<String> iterator2 = oneObject.outgoing.iterator();
				while(iterator2.hasNext()) {
					String outgoing1 = iterator2.next();
					ModelObject nextObject1 = modelObjects.get(outgoing1);
					
					Iterator<String> iterator3 = nextObject1.outgoing.iterator();
					while(iterator3.hasNext()) {
						String nextOutgoing1 = iterator3.next();
						ModelObject nextNextObject1 = modelObjects.get(nextOutgoing1);

						Iterator<String> iterator4 = nextObject1.outgoing.iterator();
						while(iterator4.hasNext()) {
							String nextOutgoing2 = iterator4.next();
							ModelObject nextNextObject2 = modelObjects.get(nextOutgoing2);
							
							if(nextNextObject1.id.compareTo(nextNextObject2.id) < 0) {
								if(nextNextObject1.name.compareTo(nextNextObject2.name) <= 0) {
									if(directConcurrent[modelObjectsIndex.get(nextNextObject1.id)][modelObjectsIndex.get(nextNextObject2.id)].equals("neverConcurrent") || directConcurrent[modelObjectsIndex.get(nextNextObject1.id)][modelObjectsIndex.get(nextNextObject2.id)].equals(""))
										directConcurrent[modelObjectsIndex.get(nextNextObject1.id)][modelObjectsIndex.get(nextNextObject2.id)] = "neverConcurrent";
									else
										directConcurrent[modelObjectsIndex.get(nextNextObject1.id)][modelObjectsIndex.get(nextNextObject2.id)] = "sometimesConcurrent";
								} else {
									if(directConcurrent[modelObjectsIndex.get(nextNextObject2.id)][modelObjectsIndex.get(nextNextObject1.id)].equals("neverConcurrent") || directConcurrent[modelObjectsIndex.get(nextNextObject2.id)][modelObjectsIndex.get(nextNextObject1.id)].equals(""))
										directConcurrent[modelObjectsIndex.get(nextNextObject2.id)][modelObjectsIndex.get(nextNextObject1.id)] = "neverConcurrent";
									else
										directConcurrent[modelObjectsIndex.get(nextNextObject2.id)][modelObjectsIndex.get(nextNextObject1.id)] = "sometimesConcurrent";
								}
							}
						}
		    		}
					
					Iterator<String> iterator4 = oneObject.outgoing.iterator();
					while(iterator4.hasNext()) {
						String outgoing2 = iterator4.next();
						ModelObject nextObject2 = modelObjects.get(outgoing2);
						
						if(nextObject1.id.compareTo(nextObject2.id) < 0) {
							Iterator<String> iterator5 = nextObject1.outgoing.iterator();
							while(iterator5.hasNext()) {
								String nextOutgoing1 = iterator5.next();
								ModelObject nextNextObject1 = modelObjects.get(nextOutgoing1);
	
								Iterator<String> iterator6 = nextObject2.outgoing.iterator();
								while(iterator6.hasNext()) {
									String nextOutgoing2 = iterator6.next();
									ModelObject nextNextObject2 = modelObjects.get(nextOutgoing2);
									
									if(nextObject1.outgoing.size() == 1 && nextObject2.outgoing.size() == 1) {
										if(nextNextObject1.name.compareTo(nextNextObject2.name) <= 0) {
											if(directConcurrent[modelObjectsIndex.get(nextNextObject1.id)][modelObjectsIndex.get(nextNextObject2.id)].equals("alwaysConcurrent") || directConcurrent[modelObjectsIndex.get(nextNextObject1.id)][modelObjectsIndex.get(nextNextObject2.id)].equals(""))
												directConcurrent[modelObjectsIndex.get(nextNextObject1.id)][modelObjectsIndex.get(nextNextObject2.id)] = "alwaysConcurrent";
											else
												directConcurrent[modelObjectsIndex.get(nextNextObject1.id)][modelObjectsIndex.get(nextNextObject2.id)] = "sometimesConcurrent";
										} else {
											if(directConcurrent[modelObjectsIndex.get(nextNextObject2.id)][modelObjectsIndex.get(nextNextObject1.id)].equals("alwaysConcurrent") || directConcurrent[modelObjectsIndex.get(nextNextObject2.id)][modelObjectsIndex.get(nextNextObject1.id)].equals(""))
												directConcurrent[modelObjectsIndex.get(nextNextObject2.id)][modelObjectsIndex.get(nextNextObject1.id)] = "alwaysConcurrent";
											else
												directConcurrent[modelObjectsIndex.get(nextNextObject2.id)][modelObjectsIndex.get(nextNextObject1.id)] = "sometimesConcurrent";
										}
									} else {
										if(nextNextObject1.name.compareTo(nextNextObject2.name) <= 0) {
											directConcurrent[modelObjectsIndex.get(nextNextObject1.id)][modelObjectsIndex.get(nextNextObject2.id)] = "sometimesConcurrent";
										} else {
											directConcurrent[modelObjectsIndex.get(nextNextObject2.id)][modelObjectsIndex.get(nextNextObject1.id)] = "sometimesConcurrent";
										}
									}
								}
							}
						}
					}
	    		}
    		}
        }

		// delete VerticalEmptyTransition
		iterator = modelObjects.entrySet().iterator();
		while(iterator.hasNext()) {
			Map.Entry<String, ModelObject> object = iterator.next();
    		ModelObject oneObject = object.getValue();
    		
    		if(oneObject.type.equals("VerticalEmptyTransition") || (oneObject.type.equals("Transition") && oneObject.name.equals(""))) {
		    	for(int i = 0; i < modelObjectsIndexCount; i++)
		    		for(int j = 0; j < modelObjectsIndexCount; j++) {
		    			if(directCausal[i][modelObjectsIndex.get(oneObject.id)].equals("directAlwaysCausal") && directCausal[modelObjectsIndex.get(oneObject.id)][j].equals("directAlwaysCausal")) {
		    				if(directCausal[i][j].equals("directSometimesCausal"))
		    					directCausal[i][j] = "directSometimesCausal";
		    				else
		    					directCausal[i][j] = "directAlwaysCausal";
		    			} else if(!directCausal[i][modelObjectsIndex.get(oneObject.id)].equals("") && !directCausal[modelObjectsIndex.get(oneObject.id)][j].equals(""))
		    				directCausal[i][j] = "directSometimesCausal";
		    			
		    			if(directInverseCausal[i][modelObjectsIndex.get(oneObject.id)].equals("directAlwaysInverseCausal") && directInverseCausal[modelObjectsIndex.get(oneObject.id)][j].equals("directAlwaysInverseCausal")) {
		    				if(directInverseCausal[i][j].equals("directSometimesInverseCausal"))
		    					directInverseCausal[i][j] = "directSometimesInverseCausal";
		    				else
		    					directInverseCausal[i][j] = "directAlwaysInverseCausal";
		    			} else if(!directInverseCausal[i][modelObjectsIndex.get(oneObject.id)].equals("") && !directInverseCausal[modelObjectsIndex.get(oneObject.id)][j].equals(""))
		    				directInverseCausal[i][j] = "directSometimesInverseCausal";
		    			
		    			if((directConcurrent[i][modelObjectsIndex.get(oneObject.id)].equals("alwaysConcurrent") || directConcurrent[modelObjectsIndex.get(oneObject.id)][i].equals("alwaysConcurrent")) && directCausal[modelObjectsIndex.get(oneObject.id)][j].equals("directAlwaysCausal")) {
		    				if(modelObjects.get(modelObjectsInverseIndex.get(i)).name.compareTo(modelObjects.get(modelObjectsInverseIndex.get(j)).name) <= 0) {
								if(directConcurrent[i][j].equals("alwaysConcurrent") || directConcurrent[i][j].equals(""))
									directConcurrent[i][j] = "alwaysConcurrent";
								else
									directConcurrent[i][j] = "sometimesConcurrent";
							} else {
								if(directConcurrent[j][i].equals("alwaysConcurrent") || directConcurrent[j][i].equals(""))
									directConcurrent[j][i] = "alwaysConcurrent";
								else
									directConcurrent[j][i] = "sometimesConcurrent";
							}
		    			} else if((directConcurrent[i][modelObjectsIndex.get(oneObject.id)].equals("neverConcurrent") || directConcurrent[modelObjectsIndex.get(oneObject.id)][i].equals("neverConcurrent")) && !directCausal[modelObjectsIndex.get(oneObject.id)][j].equals("")) {
		    				if(modelObjects.get(modelObjectsInverseIndex.get(i)).name.compareTo(modelObjects.get(modelObjectsInverseIndex.get(j)).name) <= 0) {
								if(directConcurrent[i][j].equals("neverConcurrent") || directConcurrent[i][j].equals(""))
									directConcurrent[i][j] = "neverConcurrent";
								else
									directConcurrent[i][j] = "sometimesConcurrent";
							} else {
								if(directConcurrent[j][i].equals("neverConcurrent") || directConcurrent[j][i].equals(""))
									directConcurrent[j][i] = "neverConcurrent";
								else
									directConcurrent[j][i] = "sometimesConcurrent";
							}
		    			} else if(((directConcurrent[i][modelObjectsIndex.get(oneObject.id)].equals("sometimesConcurrent") || directConcurrent[modelObjectsIndex.get(oneObject.id)][i].equals("sometimesConcurrent")) && !directCausal[modelObjectsIndex.get(oneObject.id)][j].equals("")) || ((directConcurrent[i][modelObjectsIndex.get(oneObject.id)].equals("alwaysConcurrent") || directConcurrent[modelObjectsIndex.get(oneObject.id)][i].equals("alwaysConcurrent")) && directCausal[modelObjectsIndex.get(oneObject.id)][j].equals("directSometimesCausal"))) {
		    				if(modelObjects.get(modelObjectsInverseIndex.get(i)).name.compareTo(modelObjects.get(modelObjectsInverseIndex.get(j)).name) <= 0) {
								directConcurrent[i][j] = "sometimesConcurrent";
							} else {
								directConcurrent[j][i] = "sometimesConcurrent";
							}
		    			}
		    		}
    		}
        }

		// write relationship
		try {
	        FileWriter fw = new FileWriter(filename.replaceAll("_my_petri", "_my_relationship"));
	        PrintWriter out = new PrintWriter(fw);
	    	for(int i = 0; i < modelObjectsIndexCount; i++) {
	    		ModelObject oneObject = modelObjects.get(modelObjectsInverseIndex.get(i));
	    		if(oneObject.type.equals("Transition") && !oneObject.name.equals("")) {
		    		for(int j = 0; j < modelObjectsIndexCount; j++) {
		        		ModelObject anotherObject = modelObjects.get(modelObjectsInverseIndex.get(j));
		        		if(anotherObject.type.equals("Transition") && !anotherObject.name.equals("")) {
					        out.write(oneObject.name + "\t" + anotherObject.name + "\t" + directCausal[i][j] + "/" + directInverseCausal[i][j] + "/" + directConcurrent[i][j] + "\t" + filename.replaceAll("_my_petri.json", ""));
					        out.println();
		        		}
		    		}
	    		}
	    	}
	        fw.close();
	        out.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
    public static void DelFile(String path) {
        File file = new File(path);
        if(file.exists() && file.isFile())
            file.delete();
    }
    
    public static void WriteJson(String filePath, JSONObject json) throws IOException {
    	WriteFile(filePath, json.toString(4));
    }
    
    public static void WriteFile(String filePath, String sets) throws IOException {  
        FileWriter fw = new FileWriter(filePath);  
        PrintWriter out = new PrintWriter(fw);  
        out.write(sets);  
        out.println();  
        fw.close();  
        out.close();  
    }
	
    public static JSONObject ReadJson(String path) {
    	String s = ReadFile(path);
    	JSONObject jsonObject = new JSONObject(s);
    	return jsonObject;
    }
    
    public static String ReadFile(String path) {
		File file = new File(path);
		BufferedReader reader = null;
		String laststr = "";
		try {
			reader = new BufferedReader(new FileReader(file));
			String tempString = null;
			while ((tempString = reader.readLine()) != null) {
				laststr = laststr + tempString;
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
		return laststr;
    }
}
