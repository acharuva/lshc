package org.mlbio.classifier;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Vector;

import org.apache.hadoop.io.Writable;

public class PIFArray implements Writable
{
	public Vector<PIF> pifs;
    public PIFArray(){
    	pifs = new Vector<PIF>();
    }
    
	public void readFields(DataInput in) throws IOException {
		int n = in.readInt();
		pifs.clear();

		PIF a = new PIF();
		for ( int i = 0;i < n; ++i ){
			a.readFields(in);
			pifs.add( new PIF(a.index,a.value) );
		}
	}

	public void write(DataOutput out) throws IOException {
		out.writeInt(pifs.size());
		for ( PIF a : pifs ) {
			a.write(out);
		}
	}
	
	public String toString() {
		String ret = "";
		for ( PIF a : pifs ) {
			ret += a.toString(); 
		}
		return ret;
	}
}