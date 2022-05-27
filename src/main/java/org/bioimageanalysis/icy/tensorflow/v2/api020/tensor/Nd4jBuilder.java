package org.bioimageanalysis.icy.tensorflow.v2.api020.tensor;


import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.Tensor;
import org.tensorflow.types.TBool;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TFloat64;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.TUint8;
import org.tensorflow.types.family.TType;


/**
 * @author Carlos GArcia Lopez de Haro and Daniel Felipe Gonzalez Obando
 */
public final class Nd4jBuilder
{
    /**
     * Utility class.
     */
    private Nd4jBuilder()
    {
    }

    @SuppressWarnings("unchecked")
    public static INDArray build(Tensor<? extends TType> tensor) throws IllegalArgumentException
    {
		switch (tensor.dataType().name())
        {
            case TBool.NAME:
                return buildFromTensorBool((Tensor<TBool>) tensor);
            case TUint8.NAME:
                return buildFromTensorByte((Tensor<TUint8>) tensor);
            case TInt32.NAME:
                return buildFromTensorInt((Tensor<TInt32>) tensor);
            case TFloat32.NAME:
                return buildFromTensorFloat((Tensor<TFloat32>) tensor);
            case TFloat64.NAME:
                return buildFromTensorDouble((Tensor<TFloat64>) tensor);
            case TInt64.NAME:
                return buildFromTensorLong((Tensor<TInt64>) tensor);
            default:
                throw new IllegalArgumentException("Unsupported tensor type: " + tensor.dataType().name());
        }
    }

    private static INDArray buildFromTensorBool(Tensor<TBool> tensor)
    {
		long[] tensorShape = tensor.shape().asArray();
		long size = 1;
		for (long ss : tensorShape) {size *= ss;}
		byte[] flatImageArray = new byte[(int) size];
		// Copy data from tensor to array
        tensor.rawData().read(flatImageArray);
		return Nd4j.create(flatImageArray, tensorShape, DataType.BOOL);
    }

    private static INDArray buildFromTensorByte(Tensor<TUint8> tensor)
    {
		long[] tensorShape = tensor.shape().asArray();
		long size = 1;
		for (long ss : tensorShape) {size *= ss;}
		byte[] flatImageArray = new byte[(int) size];
		// Copy data from tensor to array
        tensor.rawData().read(flatImageArray);
		return Nd4j.create(flatImageArray, tensorShape, DataType.INT8);
    }

    private static INDArray buildFromTensorInt(Tensor<TInt32> tensor)
    {
		long[] tensorShape = tensor.shape().asArray();
		long size = 1;
		for (long ss : tensorShape) {size *= ss;}
		int[] flatImageArray = new int[(int) size];
		// Copy data from tensor to array
        tensor.rawData().asInts().read(flatImageArray);
		return Nd4j.create(flatImageArray, tensorShape, DataType.INT32);
    }

    private static INDArray buildFromTensorFloat(Tensor<TFloat32> tensor)
    {
		long[] tensorShape = tensor.shape().asArray();
		long size = 1;
		for (long ss : tensorShape) {size *= ss;}
		float[] flatImageArray = new float[(int) size];
		// Copy data from tensor to array
        tensor.rawData().asFloats().read(flatImageArray);
		return Nd4j.create(flatImageArray, tensorShape, DataType.FLOAT);
    }

    private static INDArray buildFromTensorDouble(Tensor<TFloat64> tensor)
    {
		long[] tensorShape = tensor.shape().asArray();
		long size = 1;
		for (long ss : tensorShape) {size *= ss;}
		double[] flatImageArray = new double[(int) size];
		// Copy data from tensor to array
        tensor.rawData().asDoubles().read(flatImageArray);
		return Nd4j.create(flatImageArray, tensorShape, DataType.DOUBLE);
    }

    private static INDArray buildFromTensorLong(Tensor<TInt64> tensor)
    {
		long[] tensorShape = tensor.shape().asArray();
		long size = 1;
		for (long ss : tensorShape) {size *= ss;}
		long[] flatImageArray = new long[(int) size];
		// Copy data from tensor to array
        tensor.rawData().asLongs().read(flatImageArray);
		return Nd4j.create(flatImageArray, tensorShape, DataType.INT64);
    }
}
