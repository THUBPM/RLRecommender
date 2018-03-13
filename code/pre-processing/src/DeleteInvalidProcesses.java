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
import java.util.Map.Entry;
import java.util.Set;
import java.util.UUID;

import org.json.JSONArray;
import org.json.JSONObject;

public class DeleteInvalidProcesses {
	static String root = "..\\models_en";
//	static String root = "..\\models_en\\bpmn2.0\\2081504877\\BPMN2.0_Process\\2012-02-10_Old Process";
	static boolean coverOldFile = false;
	
	public static class JsonStatistics {
		public int activityCount = 0;
		public int totalLength = 0;
		
		public JsonStatistics() {
			super();
			this.activityCount = 0;
			this.totalLength = 0;
		}
	}
	
	static int count = 0;
	static Map<String, Integer> activityMap = new HashMap<String, Integer>();
   
	public static void main(String[] args) {
    	File file = new File(root);
    	GetDirectory(file);
    	System.out.println("count==>" + count);
    	System.out.println("activity==>" + activityMap.size());
    	WriteActivityCount("activityCount.txt");
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
						
						// only last version
//						String isLastRevision = metadata.getJSONObject("revision").getString("isLastRevision");
//						if(!isLastRevision.equals("true"))
//							continue;
						
						// only 100% connected models
//						int connectedness = metadata.getJSONObject("revision").getInt("connectedness");
//						if(!(connectedness == 100))
//							continue;
						
						// count
						count++;
						if(count % 100 == 0) {
							System.out.println("count==>" + count);
						}

						// get old json
						JSONObject json = ReadJson(f.getParent() + "\\" + fileName.replaceAll("_metadata", ""));
						
						// delete non-english models
						String language = metadata.getJSONObject("model").getString("naturalLanguage");
						if(!language.equals("en")) {
							DelFile(f.getParent() + "\\" + fileName.replaceAll("_metadata", ""));
							DelFile(f.getParent() + "\\" + fileName.replaceAll("_metadata.json", ".svg"));
							DelFile(f.getParent() + "\\" + fileName);
							
							continue;
						}
						
						// write json in a new format for each model
						File newJsonFile = new File(f.getParent() + "\\" + fileName.replaceAll("_metadata", "_my_new"));
						if(coverOldFile || !newJsonFile.exists()) {
							boolean isPetriNet = metadata.getJSONObject("model").getString("groupName").equals("PetriNet");
							JSONArray extractResultArray = new JSONArray();
							GenerateJson(json, isPetriNet, extractResultArray, null);
							JSONObject extractResult = new JSONObject();
							extractResult.put("list", extractResultArray);
							WriteJson(f.getParent() + "\\" + fileName.replaceAll("_metadata", "_my_new"), extractResult);
						}
						
						CountActivities(json);
						
						// delete models with few activities or short activity string length
				    	JsonStatistics result = new JsonStatistics();
				    	GetStatistics(json, result);
				    	if(result.activityCount <= 1 || 1.0 * result.totalLength / result.activityCount < 3) {
							DelFile(f.getParent() + "\\" + fileName.replaceAll("_metadata", ""));
							DelFile(f.getParent() + "\\" + fileName.replaceAll("_metadata.json", ".svg"));
							DelFile(f.getParent() + "\\" + fileName);
							DelFile(f.getParent() + "\\" + fileName.replaceAll("_metadata", "_my_new"));
							
							continue;
						}
					}
			    }
			} catch (Exception e) {
				e.printStackTrace();
				System.out.println("File==>" + f.getAbsolutePath()); 
			}
		}
	}
    
	// generate json array in a new format
    public static void GenerateJson(JSONObject json, boolean isPetriNet, JSONArray result, JSONArray parent) {
    	JSONArray childShapes = json.getJSONArray("childShapes");
    	for(int i = 0; i < childShapes.length(); i++) {
    		JSONObject childShape = childShapes.getJSONObject(i);
    		
			if(!(parent == null)) {
				parent.put(childShape.getString("resourceId"));
			}
			
			JSONObject object = new JSONObject();
			
			object.put("type", childShape.getJSONObject("stencil").getString("id"));
			if(childShape.getJSONObject("properties").has("looptype") && !childShape.getJSONObject("properties").get("looptype").equals("None")) {
				object.put("loop", childShape.getJSONObject("properties").get("looptype"));
			} else
				object.put("loop", "None");
			object.put("id", childShape.getString("resourceId"));
			if(!isPetriNet && childShape.getJSONObject("properties").has("name")) {
				object.put("name", childShape.getJSONObject("properties").getString("name").toLowerCase().replaceAll("\\s*", "").replaceAll("glossary://[0-9a-z]*/", "").replaceAll("[^a-z]", ""));
			} else if(isPetriNet && childShape.getJSONObject("properties").has("title")) {
				object.put("name", childShape.getJSONObject("properties").getString("title").toLowerCase().replaceAll("\\s*", "").replaceAll("glossary://[0-9a-z]*/", "").replaceAll("[^a-z]", ""));
			} else {
				object.put("name", "");
			}
			
	    	JSONArray outgoings = childShape.getJSONArray("outgoing");
	    	JSONArray outIds = new JSONArray();
	    	for(int j = 0; j < outgoings.length(); j++) {
	    		JSONObject outgoing = outgoings.getJSONObject(j);
	    		outIds.put(outgoing.getString("resourceId"));
	    	}
	    	object.put("outgoing", outIds);
	    	
	    	JSONArray contains = new JSONArray();
	    	object.put("contains", contains);
	    	
    		if(childShape.getJSONArray("childShapes").length() > 0) {
    			GenerateJson(childShape, isPetriNet, result, contains);
   			}
    		
    		result.put(object);
    	}
    }
	
    // count activities
    public static void CountActivities(JSONObject json) {
    	JSONArray childShapes = json.getJSONArray("childShapes");
    	for(int i = 0; i < childShapes.length(); i++) {
    		JSONObject childShape = childShapes.getJSONObject(i);
    		if(childShape.getJSONArray("childShapes").length() > 0) {
    			CountActivities(childShape);
   			} else {
    			String stencil = childShape.getJSONObject("stencil").getString("id");
    			
    			if(stencil.equals("Task")) {
    				String key = childShape.getJSONObject("properties").getString("name").toLowerCase().replaceAll("\\s*", "").replaceAll("glossary://[0-9a-z]*/", "").replaceAll("[^a-z]", "");
    				if(activityMap.containsKey(key)) {
	    				int tempint = activityMap.get(key);
	    				activityMap.put(key, tempint+1);
    				} else {
    					activityMap.put(key, 1);
    				}
    			} else if(stencil.equals("Transition")) {
					String key = childShape.getJSONObject("properties").getString("title").toLowerCase().replaceAll("\\s*", "").replaceAll("glossary://[0-9a-z]*/", "").replaceAll("[^a-z]", "");
					if(activityMap.containsKey(key)) {
	    				int tempint = activityMap.get(key);
	    				activityMap.put(key, tempint+1);
    				} else {
    					activityMap.put(key, 1);
    				}
    			}
    		}
    	}
    }
	
    // get statistics (total activity length, count...)
    public static JsonStatistics GetStatistics(JSONObject json, JsonStatistics result) {
    	JSONArray childShapes = json.getJSONArray("childShapes");
    	for(int i = 0; i < childShapes.length(); i++) {
    		JSONObject childShape = childShapes.getJSONObject(i);
    		if(childShape.getJSONArray("childShapes").length() > 0) {
    			GetStatistics(childShape, result);
    		} else {
    			String stencil = childShape.getJSONObject("stencil").getString("id");
    			if(stencil.equals("Task")) {
    				String key = childShape.getJSONObject("properties").getString("name").toLowerCase().replaceAll("\\s*", "").replaceAll("glossary://[0-9a-z]*/", "").replaceAll("[^a-z]", "");
    				result.totalLength += key.length();
    				result.activityCount++;
    			} else if(stencil.equals("Transition")) {
    				String key = childShape.getJSONObject("properties").getString("title").toLowerCase().replaceAll("\\s*", "").replaceAll("glossary://[0-9a-z]*/", "").replaceAll("[^a-z]", "");
    				result.totalLength += key.length();
    				result.activityCount++;
    			}
    		}
    	}

    	return result;
    }
	
    // write acitivty count file
    public static void WriteActivityCount(String path) {
    	try {
	    	FileWriter file = new FileWriter(path);
	    	
	        List<Map.Entry<String, Integer>> list = new ArrayList<Map.Entry<String, Integer>>(activityMap.entrySet());
	        Collections.sort(list, new Comparator<Map.Entry<String, Integer>>() {
	            public int compare(Entry<String, Integer> o1, Entry<String, Integer> o2) {
	            	if(o1.getKey().length() < o2.getKey().length())
	            		return 1;
	            	if(o1.getKey().length() > o2.getKey().length())
	            		return -1;
	            	return o1.getKey().compareTo(o2.getKey());
	            }
	        });
	        
	        for(Map.Entry<String, Integer> mapping : list) {  
	        	file.write(mapping.getKey() + " " + mapping.getValue() + "\n");
	        }
	        
	        file.close();
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
