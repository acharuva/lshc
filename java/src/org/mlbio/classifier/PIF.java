package org.mlbio.classifier;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import org.apache.hadoop.io.WritableComparable;



/* 
 * Just index and value pair, compared based on value 
 * This and the PIFArray class are not used in the main code 
 * only in TestingDriver.java
 * not even in TrainingDriver.java
 */
public class PIF implements WritableComparable<PIF> {
	public int index = 0;
	public float value = 0;
	
	public PIF ( ){		
	}
	
	public PIF ( PIF a ) {
		index = a.index;
		value = a.value;
	}
	
	public PIF ( int _index , float _value ) {
		index = _index;
		value = _value;
	}

	public void readFields(DataInput in) throws IOException {
		index = in.readInt();
		value = in.readFloat();
	}

	public void write(DataOutput out) throws IOException {
		out.writeInt( index );
		out.writeFloat( value );		
	}
	
	public String toString(){
		return index+":"+value+" ";
	}

	public int compareTo(PIF other) {
		if ( value < other.value ) return -1;
		if ( value == other.value ) return 0;
		else return +1;
	}
	
}