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

public class TransformToPetriNet_silent {
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
   
	public static void main(String[] args) {
    	File file = new File(root);
    	GetDirectory(file);
    	System.out.println("count==>" + count);
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
						
						// count
						count++;
						if(count % 100 == 0) {
							System.out.println("count==>" + count);
						}
						
						// filter new json
						File petriNetFile = new File(f.getParent() + "\\" + fileName.replaceAll("_metadata", "_my_petri"));
						if(coverOldFile || !petriNetFile.exists()) {
							boolean isPetriNet = metadata.getJSONObject("model").getString("groupName").equals("PetriNet");
							JSONObject newJson = ReadJson(f.getParent() + "\\" + fileName.replaceAll("_metadata", "_my_new"));
							JSONObject filteredResult = TransformBPMN(newJson, isPetriNet);
							WriteJson(f.getParent() + "\\" + fileName.replaceAll("_metadata", "_my_petri"), filteredResult);
						}
					}
			    }
			} catch (Exception e) {
				e.printStackTrace();
				System.out.println("File==>" + f.getAbsolutePath()); 
			}
		}
	}
    
	// Transform to BPMN
    public static JSONObject TransformBPMN(JSONObject json, boolean isPetriNet) {
    	class ModelObject {
    		public String name = null;
    		public String id = null;
    		public String type = null;
    		public String originalType = null;
    		public String loop = null;
    		public Set<String> contains = null;
    		public Set<String> SequenceOutgoing = null;
    		public Set<String> SequenceIngoing = null;
    		public Set<String> MessageOutgoing = null;
    		public Set<String> MessageIngoing = null;
    		public Set<String> ArcOutgoing = null;
    		public Set<String> ArcIngoing = null;
    		
    		public ModelObject() {
    			super();
    			this.name = "";
    			this.id = "new-" + UUID.randomUUID().toString();
    			this.type = "Place";
    			this.loop = "None";
    			this.originalType = "";
    			this.contains = new HashSet<String>();
    			this.SequenceOutgoing = new HashSet<String>();
    			this.SequenceIngoing = new HashSet<String>();
    			this.MessageOutgoing = new HashSet<String>();
    			this.MessageIngoing = new HashSet<String>();
    			this.ArcOutgoing = new HashSet<String>();
    			this.ArcIngoing = new HashSet<String>();
    		}
    		
    		public ModelObject(JSONObject oneObject) {
    			super();
    			this.name = oneObject.getString("name");
    			this.id = oneObject.getString("id");
    			this.type = oneObject.getString("type");
    			this.loop = oneObject.getString("loop");
    			this.originalType = oneObject.getString("type");
    			
    			this.contains = new HashSet<String>();
    			JSONArray contains = oneObject.getJSONArray("contains");
    			for(int i = 0; i < contains.length(); i++) {
    				this.contains.add(contains.getString(i));
    			}
    			
    			this.SequenceOutgoing = new HashSet<String>();
    			JSONArray sequenceOutgoings = oneObject.getJSONArray("outgoing");
    			for(int i = 0; i < sequenceOutgoings.length(); i++) {
    				this.SequenceOutgoing.add(sequenceOutgoings.getString(i));
    			}
    			
    			this.SequenceIngoing = new HashSet<String>();
    			this.MessageOutgoing = new HashSet<String>();
    			this.MessageIngoing = new HashSet<String>();
    			this.ArcOutgoing = new HashSet<String>();
    			this.ArcIngoing = new HashSet<String>();
    		}
    	}
    	
    	// load json
    	Map<String, ModelObject> modelObjects = new HashMap<String, ModelObject>();
    	Map<String, ModelObject> newModelObjects = new HashMap<String, ModelObject>();
    	JSONArray list = json.getJSONArray("list");
    	for(int i = 0; i < list.length(); i++) {
    		JSONObject oneObject = list.getJSONObject(i);
			modelObjects.put(oneObject.getString("id"), new ModelObject(oneObject));
			newModelObjects.put(oneObject.getString("id"), new ModelObject(oneObject));
    	}
    	
    	// add ingoings
		Iterator<Map.Entry<String, ModelObject>> iterator = modelObjects.entrySet().iterator();
		while(iterator.hasNext()) {
			Map.Entry<String, ModelObject> object = iterator.next();
    		ModelObject oneObject = newModelObjects.get(object.getKey());
    		
    		Iterator<String> iterator2 = oneObject.SequenceOutgoing.iterator();
			while(iterator2.hasNext()) {
				String sequenceOutgoing = iterator2.next();
    			if(newModelObjects.containsKey(sequenceOutgoing)) {
    				newModelObjects.get(sequenceOutgoing).SequenceIngoing.add(oneObject.id);
    			} else
    				iterator2.remove();
    		}
        }
    	
    	if(isPetriNet) {
    		// delete arcs
    		iterator = modelObjects.entrySet().iterator();
    		while(iterator.hasNext()) {
    			Map.Entry<String, ModelObject> object = iterator.next();
    			ModelObject oneObject = newModelObjects.get(object.getKey());
    			
	    		if(oneObject.type.equals("Arc")) {
	        		for (String ingoing : oneObject.SequenceIngoing) {
		        		for (String outgoing : oneObject.SequenceOutgoing) {
		        			newModelObjects.get(ingoing).ArcOutgoing.add(outgoing);
		        			newModelObjects.get(outgoing).ArcIngoing.add(ingoing);
		        		}
	        		}
	        		for (String ingoing : oneObject.SequenceIngoing) {
	        			newModelObjects.get(ingoing).SequenceOutgoing.remove(oneObject.id);
	        		}
	        		for (String outgoing : oneObject.SequenceOutgoing) {
	        			newModelObjects.get(outgoing).SequenceIngoing.remove(oneObject.id);
	        		}
	        		newModelObjects.remove(oneObject.id);
	        		iterator.remove();
	    		}
	        }
    	} else {
    		// delete edges and some objects
    		iterator = modelObjects.entrySet().iterator();
    		while(iterator.hasNext()) {
    			Map.Entry<String, ModelObject> object = iterator.next();
    			ModelObject oneObject = newModelObjects.get(object.getKey());
    			
	    		if(oneObject.type.equals("Group") || oneObject.type.equals("TextAnnotation")
	    				|| oneObject.type.equals("VerticalLane") || oneObject.type.equals("Lane")
	    				|| oneObject.type.equals("CollapsedVerticalPool") || oneObject.type.equals("Pool")
	    				|| oneObject.type.equals("CollapsedPool") || oneObject.type.equals("VerticalPool")) {
	        		for (String sequenceIngoing : oneObject.SequenceIngoing) {
	        			newModelObjects.get(sequenceIngoing).SequenceOutgoing.remove(oneObject.id);
	        		}
	        		for (String sequenceOutgoing : oneObject.SequenceOutgoing) {
	        			newModelObjects.get(sequenceOutgoing).SequenceIngoing.remove(oneObject.id);
	        		}
	        		for (String messageIngoing : oneObject.MessageIngoing) {
	        			newModelObjects.get(messageIngoing).MessageOutgoing.remove(oneObject.id);
	        		}
	        		for (String messageOutgoing : oneObject.MessageOutgoing) {
	        			newModelObjects.get(messageOutgoing).MessageIngoing.remove(oneObject.id);
	        		}
	        		newModelObjects.remove(oneObject.id);
	        		iterator.remove();
	    		}
	    		else if(oneObject.type.equals("SequenceFlow")) {
	        		for (String sequenceIngoing : oneObject.SequenceIngoing) {
		        		for (String sequenceOutgoing : oneObject.SequenceOutgoing) {
		        			newModelObjects.get(sequenceIngoing).SequenceOutgoing.add(sequenceOutgoing);
		        			newModelObjects.get(sequenceOutgoing).SequenceIngoing.add(sequenceIngoing);
		        		}
	        		}
	        		for (String messageIngoing : oneObject.MessageIngoing) {
		        		for (String messageOutgoing : oneObject.MessageOutgoing) {
		        			newModelObjects.get(messageIngoing).MessageOutgoing.add(messageOutgoing);
		        			newModelObjects.get(messageOutgoing).MessageIngoing.add(messageIngoing);
		        		}
	        		}
	        		for (String sequenceIngoing : oneObject.SequenceIngoing) {
		        		for (String messageOutgoing : oneObject.MessageOutgoing) {
		        			newModelObjects.get(sequenceIngoing).MessageOutgoing.add(messageOutgoing);
		        			newModelObjects.get(messageOutgoing).MessageIngoing.add(sequenceIngoing);
		        		}
	        		}
	        		for (String messageIngoing : oneObject.MessageIngoing) {
		        		for (String sequenceOutgoing : oneObject.SequenceOutgoing) {
		        			newModelObjects.get(messageIngoing).MessageOutgoing.add(sequenceOutgoing);
		        			newModelObjects.get(sequenceOutgoing).MessageIngoing.add(messageIngoing);
		        		}
	        		}
	        		for (String sequenceIngoing : oneObject.SequenceIngoing) {
	        			newModelObjects.get(sequenceIngoing).SequenceOutgoing.remove(oneObject.id);
	        		}
	        		for (String sequenceOutgoing : oneObject.SequenceOutgoing) {
	        			newModelObjects.get(sequenceOutgoing).SequenceIngoing.remove(oneObject.id);
	        		}
	        		for (String messageIngoing : oneObject.MessageIngoing) {
	        			newModelObjects.get(messageIngoing).MessageOutgoing.remove(oneObject.id);
	        		}
	        		for (String messageOutgoing : oneObject.MessageOutgoing) {
	        			newModelObjects.get(messageOutgoing).MessageIngoing.remove(oneObject.id);
	        		}
	        		newModelObjects.remove(oneObject.id);
	        		iterator.remove();
	    		}
	    		else if(oneObject.type.equals("MessageFlow") || oneObject.type.equals("Association_Unidirectional") || oneObject.type.equals("Association_Undirected")) {
	        		for (String sequenceIngoing : oneObject.SequenceIngoing) {
		        		for (String sequenceOutgoing : oneObject.SequenceOutgoing) {
		        			newModelObjects.get(sequenceIngoing).MessageOutgoing.add(sequenceOutgoing);
		        			newModelObjects.get(sequenceOutgoing).MessageIngoing.add(sequenceIngoing);
		        		}
	        		}
	        		for (String messageIngoing : oneObject.MessageIngoing) {
		        		for (String messageOutgoing : oneObject.MessageOutgoing) {
		        			newModelObjects.get(messageIngoing).MessageOutgoing.add(messageOutgoing);
		        			newModelObjects.get(messageOutgoing).MessageIngoing.add(messageIngoing);
		        		}
	        		}
	        		for (String sequenceIngoing : oneObject.SequenceIngoing) {
		        		for (String messageOutgoing : oneObject.MessageOutgoing) {
		        			newModelObjects.get(sequenceIngoing).MessageOutgoing.add(messageOutgoing);
		        			newModelObjects.get(messageOutgoing).MessageIngoing.add(sequenceIngoing);
		        		}
	        		}
	        		for (String messageIngoing : oneObject.MessageIngoing) {
		        		for (String sequenceOutgoing : oneObject.SequenceOutgoing) {
		        			newModelObjects.get(messageIngoing).MessageOutgoing.add(sequenceOutgoing);
		        			newModelObjects.get(sequenceOutgoing).MessageIngoing.add(messageIngoing);
		        		}
	        		}
	        		for (String sequenceIngoing : oneObject.SequenceIngoing) {
	        			newModelObjects.get(sequenceIngoing).SequenceOutgoing.remove(oneObject.id);
	        		}
	        		for (String sequenceOutgoing : oneObject.SequenceOutgoing) {
	        			newModelObjects.get(sequenceOutgoing).SequenceIngoing.remove(oneObject.id);
	        		}
	        		for (String messageIngoing : oneObject.MessageIngoing) {
	        			newModelObjects.get(messageIngoing).MessageOutgoing.remove(oneObject.id);
	        		}
	        		for (String messageOutgoing : oneObject.MessageOutgoing) {
	        			newModelObjects.get(messageOutgoing).MessageIngoing.remove(oneObject.id);
	        		}
	        		newModelObjects.remove(oneObject.id);
	        		iterator.remove();
	    		}
	    		else if(oneObject.type.equals("Association_Bidirectional")) {
	        		for (String sequenceIngoing : oneObject.SequenceIngoing) {
		        		for (String sequenceOutgoing : oneObject.SequenceOutgoing) {
		        			newModelObjects.get(sequenceIngoing).MessageOutgoing.add(sequenceOutgoing);
		        			newModelObjects.get(sequenceIngoing).MessageIngoing.add(sequenceOutgoing);
		        			newModelObjects.get(sequenceOutgoing).MessageOutgoing.add(sequenceIngoing);
		        			newModelObjects.get(sequenceOutgoing).MessageIngoing.add(sequenceIngoing);
		        		}
	        		}
	        		for (String messageIngoing : oneObject.MessageIngoing) {
		        		for (String messageOutgoing : oneObject.MessageOutgoing) {
		        			newModelObjects.get(messageIngoing).MessageOutgoing.add(messageOutgoing);
		        			newModelObjects.get(messageIngoing).MessageIngoing.add(messageOutgoing);
		        			newModelObjects.get(messageOutgoing).MessageOutgoing.add(messageIngoing);
		        			newModelObjects.get(messageOutgoing).MessageIngoing.add(messageIngoing);
		        		}
	        		}
	        		for (String sequenceIngoing : oneObject.SequenceIngoing) {
		        		for (String messageOutgoing : oneObject.MessageOutgoing) {
		        			newModelObjects.get(sequenceIngoing).MessageOutgoing.add(messageOutgoing);
		        			newModelObjects.get(sequenceIngoing).MessageIngoing.add(messageOutgoing);
		        			newModelObjects.get(messageOutgoing).MessageOutgoing.add(sequenceIngoing);
		        			newModelObjects.get(messageOutgoing).MessageIngoing.add(sequenceIngoing);
		        		}
	        		}
	        		for (String messageIngoing : oneObject.MessageIngoing) {
		        		for (String sequenceOutgoing : oneObject.SequenceOutgoing) {
		        			newModelObjects.get(messageIngoing).MessageOutgoing.add(sequenceOutgoing);
		        			newModelObjects.get(messageIngoing).MessageIngoing.add(sequenceOutgoing);
		        			newModelObjects.get(sequenceOutgoing).MessageOutgoing.add(messageIngoing);
		        			newModelObjects.get(sequenceOutgoing).MessageIngoing.add(messageIngoing);
		        		}
	        		}
	        		for (String sequenceIngoing : oneObject.SequenceIngoing) {
	        			newModelObjects.get(sequenceIngoing).SequenceOutgoing.remove(oneObject.id);
	        		}
	        		for (String sequenceOutgoing : oneObject.SequenceOutgoing) {
	        			newModelObjects.get(sequenceOutgoing).SequenceIngoing.remove(oneObject.id);
	        		}
	        		for (String messageIngoing : oneObject.MessageIngoing) {
	        			newModelObjects.get(messageIngoing).MessageOutgoing.remove(oneObject.id);
	        		}
	        		for (String messageOutgoing : oneObject.MessageOutgoing) {
	        			newModelObjects.get(messageOutgoing).MessageIngoing.remove(oneObject.id);
	        		}
	        		newModelObjects.remove(oneObject.id);
	        		iterator.remove();
	    		}
	        }
    		
    		// delete unconnected node
			iterator = modelObjects.entrySet().iterator();
			while(iterator.hasNext()) {
				Map.Entry<String, ModelObject> object = iterator.next();
				ModelObject oneObject = newModelObjects.get(object.getKey());
				
	    		if(oneObject.SequenceIngoing.size() == 0 && oneObject.SequenceOutgoing.size() == 0
	    				&& oneObject.MessageIngoing.size() == 0 && oneObject.MessageOutgoing.size() == 0
	    				&& oneObject.ArcIngoing.size() == 0 && oneObject.ArcOutgoing.size() == 0
	    				&& !oneObject.loop.equals("None")) {
	    			newModelObjects.remove(oneObject.id);
	    			iterator.remove();
	    		}
	        }
    		
    		// transfer to BPMN
    		iterator = modelObjects.entrySet().iterator();
    		while(iterator.hasNext()) {
    			Map.Entry<String, ModelObject> object = iterator.next();
    			ModelObject oneObject = newModelObjects.get(object.getKey());
    			
	    		if(oneObject.type.equals("Exclusive_Eventbased_Gateway") || oneObject.type.equals("Exclusive_Databased_Gateway")
	    				 || oneObject.type.equals("EventbasedGateway") || oneObject.type.equals("Complex_Gateway")
	    				 || oneObject.type.equals("ComplexGateway")) {
	    			oneObject.type = "Place";
	    			
		    		Iterator<String> iterator2 = oneObject.SequenceIngoing.iterator();
					while(iterator2.hasNext()) {
						String sequenceIngoing = iterator2.next();
						
	    				ModelObject inObject = new ModelObject();
	    				inObject.type = "VerticalEmptyTransition";
	    				ModelObject inPlace = new ModelObject();
	    				
	    				inObject.ArcOutgoing.add(oneObject.id);
	    				inObject.ArcIngoing.add(inPlace.id);
	    				inPlace.ArcOutgoing.add(inObject.id);
	    				inPlace.SequenceIngoing.add(sequenceIngoing);
		    			newModelObjects.get(sequenceIngoing).SequenceOutgoing.remove(oneObject.id);
		    			newModelObjects.get(sequenceIngoing).SequenceOutgoing.add(inPlace.id);
	        			iterator2.remove();
		    			oneObject.ArcIngoing.add(inObject.id);
		    			
		    			newModelObjects.put(inObject.id, inObject);
		    			newModelObjects.put(inPlace.id, inPlace);
		    		}
	    			
		    		iterator2 = oneObject.SequenceOutgoing.iterator();
					while(iterator2.hasNext()) {
						String sequenceOutgoing = iterator2.next();
		    			
	    				ModelObject outObject = new ModelObject();
	    				outObject.type = "VerticalEmptyTransition";
	    				ModelObject outPlace = new ModelObject();
	    				
	    				outObject.ArcIngoing.add(oneObject.id);
	    				outObject.ArcOutgoing.add(outPlace.id);
	    				outPlace.ArcIngoing.add(outObject.id);
	    				outPlace.SequenceOutgoing.add(sequenceOutgoing);
		    			newModelObjects.get(sequenceOutgoing).SequenceIngoing.remove(oneObject.id);
		    			newModelObjects.get(sequenceOutgoing).SequenceIngoing.add(outPlace.id);
	        			iterator2.remove();
		    			oneObject.ArcOutgoing.add(outObject.id);
		    			
		    			newModelObjects.put(outObject.id, outObject);
		    			newModelObjects.put(outPlace.id, outPlace);
		    		}
	    		}
	    		else if(oneObject.type.equals("ParallelGateway") || oneObject.type.equals("AND_Gateway")) {
	    			oneObject.type = "VerticalEmptyTransition";
	    			
		    		Iterator<String> iterator2 = oneObject.SequenceIngoing.iterator();
					while(iterator2.hasNext()) {
						String sequenceIngoing = iterator2.next();
						
	    				ModelObject inPlace = new ModelObject();
	        			inPlace.SequenceIngoing.add(sequenceIngoing);
		    			inPlace.ArcOutgoing.add(oneObject.id);
		    			newModelObjects.get(sequenceIngoing).SequenceOutgoing.remove(oneObject.id);
		    			newModelObjects.get(sequenceIngoing).SequenceOutgoing.add(inPlace.id);
	        			iterator2.remove();
		    			oneObject.ArcIngoing.add(inPlace.id);
		    			
		    			newModelObjects.put(inPlace.id, inPlace);
		    		}
	    			
		    		iterator2 = oneObject.SequenceOutgoing.iterator();
					while(iterator2.hasNext()) {
						String sequenceOutgoing = iterator2.next();
						
						ModelObject outPlace = new ModelObject();
		        		outPlace.ArcIngoing.add(oneObject.id);
	        			outPlace.SequenceOutgoing.add(sequenceOutgoing);
	        			newModelObjects.get(sequenceOutgoing).SequenceIngoing.remove(oneObject.id);
	        			newModelObjects.get(sequenceOutgoing).SequenceIngoing.add(outPlace.id);
		        		iterator2.remove();
		    			oneObject.ArcOutgoing.add(outPlace.id);
		    			
		    			newModelObjects.put(outPlace.id, outPlace);
		    		}
	    		}
	    		else if(oneObject.type.equals("InclusiveGateway") || oneObject.type.equals("OR_Gateway")) {
	    			oneObject.type = "VerticalEmptyTransition";
	    			boolean merge = oneObject.SequenceIngoing.size() > 1;
	    			boolean or_split = oneObject.SequenceOutgoing.size() > 1;
	    			
	    			Iterator<String> iterator2 = oneObject.SequenceIngoing.iterator();
					while(iterator2.hasNext()) {
						String sequenceIngoing = iterator2.next();
						
	    				ModelObject inPlace = new ModelObject();
	        			inPlace.SequenceIngoing.add(sequenceIngoing);
		    			inPlace.ArcOutgoing.add(oneObject.id);
		    			newModelObjects.get(sequenceIngoing).SequenceOutgoing.remove(oneObject.id);
		    			newModelObjects.get(sequenceIngoing).SequenceOutgoing.add(inPlace.id);
	        			iterator2.remove();
		    			oneObject.ArcIngoing.add(inPlace.id);
		    			
		    			newModelObjects.put(inPlace.id, inPlace);
		    			
		    			if(merge) {
		    				ModelObject notOneObject = new ModelObject();
		    				notOneObject.type = "VerticalEmptyTransition";
		    				notOneObject.ArcOutgoing.add(inPlace.id);
		    				inPlace.ArcIngoing.add(notOneObject.id);
		    				newModelObjects.put(notOneObject.id, notOneObject);
		    				
		    				ModelObject notOneObjectPlace = new ModelObject();
		    				notOneObjectPlace.ArcOutgoing.add(notOneObject.id);
		    				notOneObject.ArcIngoing.add(notOneObjectPlace.id);
		    				newModelObjects.put(notOneObjectPlace.id, notOneObjectPlace);
		    			}
		    		}
	    			
					iterator2 = oneObject.SequenceOutgoing.iterator();
					while(iterator2.hasNext()) {
						String sequenceOutgoing = iterator2.next();
						
						ModelObject outPlace = new ModelObject();
		        		outPlace.ArcIngoing.add(oneObject.id);
	        			outPlace.SequenceOutgoing.add(sequenceOutgoing);
	        			newModelObjects.get(sequenceOutgoing).SequenceIngoing.remove(oneObject.id);
	        			newModelObjects.get(sequenceOutgoing).SequenceIngoing.add(outPlace.id);
		        		iterator2.remove();
		    			oneObject.ArcOutgoing.add(outPlace.id);
		    			
		    			newModelObjects.put(outPlace.id, outPlace);
		    			
		    			if(or_split) {
		    				ModelObject notOneObject = new ModelObject();
		    				notOneObject.type = "VerticalEmptyTransition";
		    				notOneObject.ArcIngoing.add(outPlace.id);
		    				outPlace.ArcOutgoing.add(notOneObject.id);
		    				newModelObjects.put(notOneObject.id, notOneObject);
		    				
		    				ModelObject notOneObjectPlace = new ModelObject();
		    				notOneObjectPlace.ArcIngoing.add(notOneObject.id);
		    				notOneObject.ArcOutgoing.add(notOneObjectPlace.id);
		    				newModelObjects.put(notOneObjectPlace.id, notOneObjectPlace);
		    			}
		    		}
	    		}
	    		else if(oneObject.loop.equals("multi_instance")) {
					if(oneObject.name.equals(""))
//						oneObject.name = oneObject.type;
						oneObject.type = "VerticalEmptyTransition";
					else
						oneObject.type = "Transition";
	    			
		    		Iterator<String> iterator2 = oneObject.MessageIngoing.iterator();
					while(iterator2.hasNext()) {
						String messageIngoing = iterator2.next();
						
	    				ModelObject inPlace = new ModelObject();
	        			inPlace.MessageIngoing.add(messageIngoing);
		    			inPlace.ArcOutgoing.add(oneObject.id);
		    			newModelObjects.get(messageIngoing).MessageOutgoing.remove(oneObject.id);
		    			newModelObjects.get(messageIngoing).MessageOutgoing.add(inPlace.id);
	        			iterator2.remove();
		    			oneObject.ArcIngoing.add(inPlace.id);
		    			
		    			newModelObjects.put(inPlace.id, inPlace);
		    		}
	    			
		    		iterator2 = oneObject.MessageOutgoing.iterator();
					while(iterator2.hasNext()) {
						String messageOutgoing = iterator2.next();
						
						ModelObject outPlace = new ModelObject();
		        		outPlace.ArcIngoing.add(oneObject.id);
	        			outPlace.MessageOutgoing.add(messageOutgoing);
	        			newModelObjects.get(messageOutgoing).MessageIngoing.remove(oneObject.id);
	        			newModelObjects.get(messageOutgoing).MessageIngoing.add(outPlace.id);
		        		iterator2.remove();
		    			oneObject.ArcOutgoing.add(outPlace.id);
		    			
		    			newModelObjects.put(outPlace.id, outPlace);
		    		}
					
    				ModelObject inPlace = new ModelObject();
		    		iterator2 = oneObject.SequenceIngoing.iterator();
					while(iterator2.hasNext()) {
						String sequenceIngoing = iterator2.next();
						
	        			inPlace.SequenceIngoing.add(sequenceIngoing);
	        			newModelObjects.get(sequenceIngoing).SequenceOutgoing.remove(oneObject.id);
	        			newModelObjects.get(sequenceIngoing).SequenceOutgoing.add(inPlace.id);
	        			iterator2.remove();
		    		}
	    			inPlace.ArcOutgoing.add(oneObject.id);
	    			oneObject.ArcIngoing.add(inPlace.id);
	    			newModelObjects.put(inPlace.id, inPlace);
	    			
					ModelObject outPlace = new ModelObject();
		    		iterator2 = oneObject.SequenceOutgoing.iterator();
					while(iterator2.hasNext()) {
						String sequenceOutgoing = iterator2.next();
						
	        			outPlace.SequenceOutgoing.add(sequenceOutgoing);
	        			newModelObjects.get(sequenceOutgoing).SequenceIngoing.remove(oneObject.id);
	        			newModelObjects.get(sequenceOutgoing).SequenceIngoing.add(outPlace.id);
		        		iterator2.remove();
		    		}
	        		outPlace.ArcIngoing.add(oneObject.id);
	    			oneObject.ArcOutgoing.add(outPlace.id);
	    			newModelObjects.put(outPlace.id, outPlace);
	    			
    				ModelObject inTransition = new ModelObject();
    				inTransition.type = "VerticalEmptyTransition";
		    		iterator2 = oneObject.ArcIngoing.iterator();
					while(iterator2.hasNext()) {
						String arcIngoing = iterator2.next();
						
						inTransition.ArcIngoing.add(arcIngoing);
						newModelObjects.get(arcIngoing).ArcOutgoing.remove(oneObject.id);
						newModelObjects.get(arcIngoing).ArcOutgoing.add(inTransition.id);
	        			iterator2.remove();
		    		}
					ModelObject in1Place = new ModelObject();
					inTransition.ArcOutgoing.add(in1Place.id);
					in1Place.ArcOutgoing.add(oneObject.id);
	    			oneObject.ArcIngoing.add(in1Place.id);
	    			in1Place.ArcIngoing.add(inTransition.id);
	    			newModelObjects.put(inTransition.id, inTransition);
	    			newModelObjects.put(in1Place.id, in1Place);
	    			
					ModelObject outTransition = new ModelObject();
					outTransition.type = "VerticalEmptyTransition";
		    		iterator2 = oneObject.ArcOutgoing.iterator();
					while(iterator2.hasNext()) {
						String arcOutgoing = iterator2.next();
						
						outTransition.ArcOutgoing.add(arcOutgoing);
						newModelObjects.get(arcOutgoing).ArcIngoing.remove(oneObject.id);
						newModelObjects.get(arcOutgoing).ArcIngoing.add(outTransition.id);
		        		iterator2.remove();
		    		}
					ModelObject out1Place = new ModelObject();
					outTransition.ArcIngoing.add(out1Place.id);
					out1Place.ArcIngoing.add(oneObject.id);
	    			oneObject.ArcOutgoing.add(out1Place.id);
	    			out1Place.ArcOutgoing.add(outTransition.id);
	    			newModelObjects.put(outTransition.id, outTransition);
	    			newModelObjects.put(out1Place.id, out1Place);
	    			
	    			ModelObject duplicateTransition = new ModelObject();
					if(oneObject.name.equals(""))
//						duplicateTransition.name = oneObject.type;
						duplicateTransition.type = "VerticalEmptyTransition";
					else {
						duplicateTransition.type = "Transition";
						duplicateTransition.name = oneObject.name;
					}
					ModelObject in2Place = new ModelObject();
					inTransition.ArcOutgoing.add(in2Place.id);
					in2Place.ArcOutgoing.add(duplicateTransition.id);
					duplicateTransition.ArcIngoing.add(in2Place.id);
	    			in2Place.ArcIngoing.add(inTransition.id);
	    			ModelObject out2Place = new ModelObject();
					outTransition.ArcIngoing.add(out2Place.id);
					out2Place.ArcIngoing.add(duplicateTransition.id);
					duplicateTransition.ArcOutgoing.add(out2Place.id);
	    			out2Place.ArcOutgoing.add(outTransition.id);
	    			newModelObjects.put(in2Place.id, in2Place);
	    			newModelObjects.put(out2Place.id, out2Place);
	    			newModelObjects.put(duplicateTransition.id, duplicateTransition);
	    		}
	    		else if(!oneObject.loop.equals("None")) {
					if(oneObject.name.equals(""))
//						oneObject.name = oneObject.type;
						oneObject.type = "VerticalEmptyTransition";
					else
						oneObject.type = "Transition";
	    			
		    		Iterator<String> iterator2 = oneObject.MessageIngoing.iterator();
					while(iterator2.hasNext()) {
						String messageIngoing = iterator2.next();
						
	    				ModelObject inPlace = new ModelObject();
	        			inPlace.MessageIngoing.add(messageIngoing);
		    			inPlace.ArcOutgoing.add(oneObject.id);
		    			newModelObjects.get(messageIngoing).MessageOutgoing.remove(oneObject.id);
		    			newModelObjects.get(messageIngoing).MessageOutgoing.add(inPlace.id);
	        			iterator2.remove();
		    			oneObject.ArcIngoing.add(inPlace.id);
		    			
		    			newModelObjects.put(inPlace.id, inPlace);
		    		}
	    			
		    		iterator2 = oneObject.MessageOutgoing.iterator();
					while(iterator2.hasNext()) {
						String messageOutgoing = iterator2.next();
						
						ModelObject outPlace = new ModelObject();
		        		outPlace.ArcIngoing.add(oneObject.id);
	        			outPlace.MessageOutgoing.add(messageOutgoing);
	        			newModelObjects.get(messageOutgoing).MessageIngoing.remove(oneObject.id);
	        			newModelObjects.get(messageOutgoing).MessageIngoing.add(outPlace.id);
		        		iterator2.remove();
		    			oneObject.ArcOutgoing.add(outPlace.id);
		    			
		    			newModelObjects.put(outPlace.id, outPlace);
		    		}
					
    				ModelObject inPlace = new ModelObject();
		    		iterator2 = oneObject.SequenceIngoing.iterator();
					while(iterator2.hasNext()) {
						String sequenceIngoing = iterator2.next();
						
	        			inPlace.SequenceIngoing.add(sequenceIngoing);
	        			newModelObjects.get(sequenceIngoing).SequenceOutgoing.remove(oneObject.id);
	        			newModelObjects.get(sequenceIngoing).SequenceOutgoing.add(inPlace.id);
	        			iterator2.remove();
		    		}
	    			inPlace.ArcOutgoing.add(oneObject.id);
	    			oneObject.ArcIngoing.add(inPlace.id);
	    			newModelObjects.put(inPlace.id, inPlace);
	    			
					ModelObject outPlace = new ModelObject();
		    		iterator2 = oneObject.SequenceOutgoing.iterator();
					while(iterator2.hasNext()) {
						String sequenceOutgoing = iterator2.next();
						
	        			outPlace.SequenceOutgoing.add(sequenceOutgoing);
	        			newModelObjects.get(sequenceOutgoing).SequenceIngoing.remove(oneObject.id);
	        			newModelObjects.get(sequenceOutgoing).SequenceIngoing.add(outPlace.id);
		        		iterator2.remove();
		    		}
	        		outPlace.ArcIngoing.add(oneObject.id);
	    			oneObject.ArcOutgoing.add(outPlace.id);
	    			newModelObjects.put(outPlace.id, outPlace);
	    			
    				ModelObject inTransition = new ModelObject();
    				inTransition.type = "VerticalEmptyTransition";
		    		iterator2 = oneObject.ArcIngoing.iterator();
					while(iterator2.hasNext()) {
						String arcIngoing = iterator2.next();
						
						inTransition.ArcIngoing.add(arcIngoing);
						newModelObjects.get(arcIngoing).ArcOutgoing.remove(oneObject.id);
						newModelObjects.get(arcIngoing).ArcOutgoing.add(inTransition.id);
	        			iterator2.remove();
		    		}
					ModelObject inLoopPlace = new ModelObject();
					inTransition.ArcOutgoing.add(inLoopPlace.id);
					inLoopPlace.ArcOutgoing.add(oneObject.id);
	    			oneObject.ArcIngoing.add(inLoopPlace.id);
	    			inLoopPlace.ArcIngoing.add(inTransition.id);
	    			newModelObjects.put(inTransition.id, inTransition);
	    			newModelObjects.put(inLoopPlace.id, inLoopPlace);
	    			
					ModelObject outTransition = new ModelObject();
					outTransition.type = "VerticalEmptyTransition";
		    		iterator2 = oneObject.ArcOutgoing.iterator();
					while(iterator2.hasNext()) {
						String arcOutgoing = iterator2.next();
						
						outTransition.ArcOutgoing.add(arcOutgoing);
						newModelObjects.get(arcOutgoing).ArcIngoing.remove(oneObject.id);
						newModelObjects.get(arcOutgoing).ArcIngoing.add(outTransition.id);
		        		iterator2.remove();
		    		}
					ModelObject outLoopPlace = new ModelObject();
					outTransition.ArcIngoing.add(outLoopPlace.id);
					outLoopPlace.ArcIngoing.add(oneObject.id);
	    			oneObject.ArcOutgoing.add(outLoopPlace.id);
	    			outLoopPlace.ArcOutgoing.add(outTransition.id);
	    			newModelObjects.put(outTransition.id, outTransition);
	    			newModelObjects.put(outLoopPlace.id, outLoopPlace);
	    			
	    			ModelObject loopTransition = new ModelObject();
	    			loopTransition.type = "VerticalEmptyTransition";
	    			loopTransition.ArcIngoing.add(outLoopPlace.id);
					inLoopPlace.ArcIngoing.add(loopTransition.id);
					loopTransition.ArcOutgoing.add(inLoopPlace.id);
	    			outLoopPlace.ArcOutgoing.add(loopTransition.id);
	    			newModelObjects.put(loopTransition.id, loopTransition);
	    		}
	    		else if(oneObject.type.equals("Task") || oneObject.type.equals("Message")
	    				|| oneObject.type.equals("ITSystem") || oneObject.type.equals("DataStore")
	    				|| oneObject.type.equals("DataObject") || oneObject.type.equals("processparticipant")
						|| oneObject.type.equals("CollapsedEventSubprocess") || oneObject.type.equals("CollapsedSubprocess")
						|| oneObject.type.equals("IntermediateConditionalEvent") || oneObject.type.equals("IntermediateEscalationEvent")
						|| oneObject.type.equals("IntermediateCancelEvent") || oneObject.type.equals("IntermediateErrorEvent")
						|| oneObject.type.equals("IntermediateTimerEvent") || oneObject.type.equals("IntermediateEvent")
						|| oneObject.type.equals("StartParallelMultipleEvent") || oneObject.type.equals("StartCompensationEvent")
						|| oneObject.type.equals("StartConditionalEvent") || oneObject.type.equals("StartEscalationEvent")
						|| oneObject.type.equals("StartMultipleEvent") || oneObject.type.equals("StartMessageEvent")
						|| oneObject.type.equals("StartSignalEvent") || oneObject.type.equals("StartErrorEvent")
						|| oneObject.type.equals("StartTimerEvent") || oneObject.type.equals("StartNoneEvent") || oneObject.type.equals("StartEvent")
						|| oneObject.type.equals("EndCompensationEvent") || oneObject.type.equals("EndEscalationEvent")
						|| oneObject.type.equals("EndTerminateEvent") || oneObject.type.equals("EndMultipleEvent")
						|| oneObject.type.equals("EndMessageEvent") || oneObject.type.equals("EndCancelEvent")
						|| oneObject.type.equals("EndSignalEvent") || oneObject.type.equals("EndErrorEvent")
						|| oneObject.type.equals("EndNoneEvent") || oneObject.type.equals("EndEvent")
						|| oneObject.type.equals("EventSubprocess") || oneObject.type.equals("Subprocess")) {
					if(oneObject.name.equals(""))
//						oneObject.name = oneObject.type;
						oneObject.type = "VerticalEmptyTransition";
					else
						oneObject.type = "Transition";
	    			
		    		Iterator<String> iterator2 = oneObject.MessageIngoing.iterator();
					while(iterator2.hasNext()) {
						String messageIngoing = iterator2.next();
						
	    				ModelObject inPlace = new ModelObject();
	        			inPlace.MessageIngoing.add(messageIngoing);
		    			inPlace.ArcOutgoing.add(oneObject.id);
		    			newModelObjects.get(messageIngoing).MessageOutgoing.remove(oneObject.id);
		    			newModelObjects.get(messageIngoing).MessageOutgoing.add(inPlace.id);
	        			iterator2.remove();
		    			oneObject.ArcIngoing.add(inPlace.id);
		    			
		    			newModelObjects.put(inPlace.id, inPlace);
		    		}
	    			
		    		iterator2 = oneObject.MessageOutgoing.iterator();
					while(iterator2.hasNext()) {
						String messageOutgoing = iterator2.next();
						
						ModelObject outPlace = new ModelObject();
		        		outPlace.ArcIngoing.add(oneObject.id);
	        			outPlace.MessageOutgoing.add(messageOutgoing);
	        			newModelObjects.get(messageOutgoing).MessageIngoing.remove(oneObject.id);
	        			newModelObjects.get(messageOutgoing).MessageIngoing.add(outPlace.id);
		        		iterator2.remove();
		    			oneObject.ArcOutgoing.add(outPlace.id);
		    			
		    			newModelObjects.put(outPlace.id, outPlace);
		    		}
					
    				ModelObject inPlace = new ModelObject();
		    		iterator2 = oneObject.SequenceIngoing.iterator();
					while(iterator2.hasNext()) {
						String sequenceIngoing = iterator2.next();
						
	        			inPlace.SequenceIngoing.add(sequenceIngoing);
	        			newModelObjects.get(sequenceIngoing).SequenceOutgoing.remove(oneObject.id);
	        			newModelObjects.get(sequenceIngoing).SequenceOutgoing.add(inPlace.id);
	        			iterator2.remove();
		    		}
	    			inPlace.ArcOutgoing.add(oneObject.id);
	    			oneObject.ArcIngoing.add(inPlace.id);
	    			newModelObjects.put(inPlace.id, inPlace);
	    			
					ModelObject outPlace = new ModelObject();
		    		iterator2 = oneObject.SequenceOutgoing.iterator();
					while(iterator2.hasNext()) {
						String sequenceOutgoing = iterator2.next();
						
	        			outPlace.SequenceOutgoing.add(sequenceOutgoing);
	        			newModelObjects.get(sequenceOutgoing).SequenceIngoing.remove(oneObject.id);
	        			newModelObjects.get(sequenceOutgoing).SequenceIngoing.add(outPlace.id);
		        		iterator2.remove();
		    		}
	        		outPlace.ArcIngoing.add(oneObject.id);
	    			oneObject.ArcOutgoing.add(outPlace.id);
	    			newModelObjects.put(outPlace.id, outPlace);
				}
	    		else if(oneObject.type.equals("IntermediateParallelMultipleEventCatching")
	    				|| oneObject.type.equals("IntermediateCompensationEventCatching") || oneObject.type.equals("IntermediateCompensationEventThrowing")
	    				|| oneObject.type.equals("IntermediateEscalationEventThrowing")
	    				|| oneObject.type.equals("IntermediateMultipleEventCatching") || oneObject.type.equals("IntermediateMultipleEventThrowing")
	    				|| oneObject.type.equals("IntermediateMessageEventCatching") || oneObject.type.equals("IntermediateMessageEventThrowing")
	    				|| oneObject.type.equals("IntermediateSignalEventCatching") || oneObject.type.equals("IntermediateSignalEventThrowing")
	    				|| oneObject.type.equals("IntermediateLinkEventCatching") || oneObject.type.equals("IntermediateLinkEventThrowing")) {
					if(oneObject.name.equals(""))
//						oneObject.name = oneObject.type;
						oneObject.type = "VerticalEmptyTransition";
					else
						oneObject.type = "Transition";
	    			
		    		Iterator<String> iterator2 = oneObject.MessageIngoing.iterator();
					while(iterator2.hasNext()) {
						String messageIngoing = iterator2.next();
						
	    				ModelObject inPlace = new ModelObject();
	        			inPlace.MessageIngoing.add(messageIngoing);
		    			inPlace.ArcOutgoing.add(oneObject.id);
		    			newModelObjects.get(messageIngoing).MessageOutgoing.remove(oneObject.id);
		    			newModelObjects.get(messageIngoing).MessageOutgoing.add(inPlace.id);
	        			iterator2.remove();
		    			oneObject.ArcIngoing.add(inPlace.id);
		    			
		    			newModelObjects.put(inPlace.id, inPlace);
		    		}
	    			
		    		iterator2 = oneObject.MessageOutgoing.iterator();
					while(iterator2.hasNext()) {
						String messageOutgoing = iterator2.next();
						
						ModelObject outPlace = new ModelObject();
		        		outPlace.ArcIngoing.add(oneObject.id);
	        			outPlace.MessageOutgoing.add(messageOutgoing);
	        			newModelObjects.get(messageOutgoing).MessageIngoing.remove(oneObject.id);
	        			newModelObjects.get(messageOutgoing).MessageIngoing.add(outPlace.id);
		        		iterator2.remove();
		    			oneObject.ArcOutgoing.add(outPlace.id);
		    			
		    			newModelObjects.put(outPlace.id, outPlace);
		    		}
					
    				ModelObject inPlace = new ModelObject();
		    		iterator2 = oneObject.SequenceIngoing.iterator();
					while(iterator2.hasNext()) {
						String sequenceIngoing = iterator2.next();
						
	        			inPlace.SequenceIngoing.add(sequenceIngoing);
	        			newModelObjects.get(sequenceIngoing).SequenceOutgoing.remove(oneObject.id);
	        			newModelObjects.get(sequenceIngoing).SequenceOutgoing.add(inPlace.id);
	        			iterator2.remove();
		    		}
	    			inPlace.ArcOutgoing.add(oneObject.id);
	    			oneObject.ArcIngoing.add(inPlace.id);
	    			newModelObjects.put(inPlace.id, inPlace);
	    			
					ModelObject outPlace = new ModelObject();
		    		iterator2 = oneObject.SequenceOutgoing.iterator();
					while(iterator2.hasNext()) {
						String sequenceOutgoing = iterator2.next();
						
	        			outPlace.SequenceOutgoing.add(sequenceOutgoing);
	        			newModelObjects.get(sequenceOutgoing).SequenceIngoing.remove(oneObject.id);
	        			newModelObjects.get(sequenceOutgoing).SequenceIngoing.add(outPlace.id);
		        		iterator2.remove();
		    		}
	        		outPlace.ArcIngoing.add(oneObject.id);
	    			oneObject.ArcOutgoing.add(outPlace.id);
	    			newModelObjects.put(outPlace.id, outPlace);
				}
	        }
    	}
    	
    	// change edge to arc
		iterator = newModelObjects.entrySet().iterator();
		while(iterator.hasNext()) {
			Map.Entry<String, ModelObject> object = iterator.next();
    		ModelObject oneObject = object.getValue();
    		
    		Iterator<String> iterator2 = oneObject.SequenceIngoing.iterator();
			while(iterator2.hasNext()) {
				String sequenceIngoing = iterator2.next();
    		
    			newModelObjects.get(sequenceIngoing).SequenceOutgoing.remove(oneObject.id);
    			newModelObjects.get(sequenceIngoing).ArcOutgoing.add(oneObject.id);
    			oneObject.ArcIngoing.add(sequenceIngoing);
    			iterator2.remove();
    		}
			
    		iterator2 = oneObject.SequenceOutgoing.iterator();
			while(iterator2.hasNext()) {
				String sequenceOutgoing = iterator2.next();
    		
    			newModelObjects.get(sequenceOutgoing).SequenceIngoing.remove(oneObject.id);
    			newModelObjects.get(sequenceOutgoing).ArcIngoing.add(oneObject.id);
    			oneObject.ArcOutgoing.add(sequenceOutgoing);
    			iterator2.remove();
    		}
			
			iterator2 = oneObject.MessageIngoing.iterator();
			while(iterator2.hasNext()) {
				String messageIngoing = iterator2.next();
    		
    			newModelObjects.get(messageIngoing).MessageOutgoing.remove(oneObject.id);
    			newModelObjects.get(messageIngoing).ArcOutgoing.add(oneObject.id);
    			oneObject.ArcIngoing.add(messageIngoing);
    			iterator2.remove();
    		}
			
    		iterator2 = oneObject.MessageOutgoing.iterator();
			while(iterator2.hasNext()) {
				String messageOutgoing = iterator2.next();
    		
    			newModelObjects.get(messageOutgoing).MessageIngoing.remove(oneObject.id);
    			newModelObjects.get(messageOutgoing).ArcIngoing.add(oneObject.id);
    			oneObject.ArcOutgoing.add(messageOutgoing);
    			iterator2.remove();
    		}
        }
    	
		// delete place to place
		boolean flag = true;
		while(flag) {
			flag = false;
			iterator = newModelObjects.entrySet().iterator();
			while(iterator.hasNext()) {
				Map.Entry<String, ModelObject> object = iterator.next();
	    		ModelObject oneObject = object.getValue();
	    		
	    		if(oneObject.type.equals("Place")) {
		    		Iterator<String> iterator2 = oneObject.ArcOutgoing.iterator();
					while(iterator2.hasNext()) {
						ModelObject outObject = newModelObjects.get(iterator2.next());
						
						if(outObject.type.equals("Place")) {
							flag = true;
			        		for (String arcIngoing : oneObject.ArcIngoing) {
		        				newModelObjects.get(arcIngoing).ArcOutgoing.remove(oneObject.id);
		        				newModelObjects.get(arcIngoing).ArcOutgoing.add(outObject.id);
		        				outObject.ArcIngoing.add(arcIngoing);
			        		}
			        		for (String arcOutgoing : oneObject.ArcOutgoing) {
			        			if(!arcOutgoing.equals(outObject.id)) {
			        				newModelObjects.get(arcOutgoing).ArcIngoing.remove(oneObject.id);
			        				newModelObjects.get(arcOutgoing).ArcIngoing.add(outObject.id);
			        				outObject.ArcOutgoing.add(arcOutgoing);
			        			}
			        		}
			        		outObject.ArcIngoing.remove(oneObject.id);
			        		iterator.remove();
			        		break;
						}
		    		}
	    		}
	        }
		}
		
    	// write json result
    	JSONArray resultList = new JSONArray();
		iterator = newModelObjects.entrySet().iterator();
		while(iterator.hasNext()) {
			Map.Entry<String, ModelObject> object = iterator.next();
    		ModelObject oneObject = object.getValue();
    		
    		JSONObject resultObject = new JSONObject();
    		resultObject.put("id", oneObject.id);
    		resultObject.put("name", oneObject.name);
    		resultObject.put("type", oneObject.type);
    		resultObject.put("loop", oneObject.loop);
    		resultObject.put("originalType", oneObject.originalType);
    		
			if(oneObject.SequenceOutgoing.size() > 0 || oneObject.SequenceIngoing.size() > 0 || oneObject.MessageOutgoing.size() > 0 || oneObject.MessageIngoing.size() > 0)
				System.out.println("Error edges not processed: " + oneObject.SequenceOutgoing.size() + " " + oneObject.SequenceIngoing.size() + " " + oneObject.MessageOutgoing.size() + " " + oneObject.MessageIngoing.size());
    		
    		JSONArray outgoingList = new JSONArray();
    		for (String outgoing : oneObject.ArcOutgoing) {
    			outgoingList.put(outgoing);
    		}
    		resultObject.put("outgoing", outgoingList);
    		
    		JSONArray ingoingList = new JSONArray();
    		for (String ingoing : oneObject.ArcIngoing) {
    			ingoingList.put(ingoing);
    		}
    		resultObject.put("ingoing", ingoingList);
    		
    		JSONArray containsList = new JSONArray();
    		for (String contains : oneObject.contains) {
    			containsList.put(contains);
    		}
    		resultObject.put("contains", containsList);
    		
    		resultList.put(resultObject);
        }
    	
    	JSONObject result = new JSONObject();
    	result.put("list", resultList);
    	
    	return result;
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
