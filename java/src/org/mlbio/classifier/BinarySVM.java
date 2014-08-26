/*
 *  Largely taken from the Dual co-ordinate Descent for L2-regularized L1-SVM 
 *  developed in http://www.csie.ntu.edu.tw/~cjlin/liblinear/
 *   
 *   - Siddharth Gopal gcdart@gmail 
 */

package org.mlbio.classifier;

import java.util.Random;
import java.util.Vector;

import org.mlbio.classifier.Example;

public class BinarySVM {
    final static double EPS = 1e-12;

    public double optimize(Vector<Example> data, WeightParameter param,
            double C, double eps, int max_iter) {

        int n = param.weightvector.length;
        int N = data.size();
        for (Example E : data) {
            n = Math.max(n, E.fsize());
        }

        int[] index = new int[N];
        int[] label = new int[N];
        double[] QD = new double[N]; // QD[i] = x_i^T*x_i
        double[] alpha = new double[N];
        double PGmax_old = Double.POSITIVE_INFINITY, PGmin_old = Double.NEGATIVE_INFINITY;
        int active_size = N, npos = 0, nneg = 0;
        Random generator = new Random();

        for (int i = 0; i < N; ++i) {
            Example E = data.get(i);
            int l = -1;
            for (int lab : E.labels)
                l = (lab == param.node ? 1 : -1);

            npos += (l > 0 ? 1 : 0);
            nneg += (l < 0 ? 1 : 0);
            QD[i] = alpha[i] = 0;
            index[i] = i;
            label[i] = l;
            for (int j = 0; j < E.fids.length; ++j)
                QD[i] += E.fvals[j] * E.fvals[j];
        }
        if (npos == 0 || nneg == 0)
            return 0;
        System.out.println(" npos = " + npos + " nneg = " + nneg + "\n");

        for (int iter = 0; iter < max_iter;) {
            double PGmax_new = Double.NEGATIVE_INFINITY, PGmin_new = Double.POSITIVE_INFINITY;

            // Random permutation of the indices
            for (int i = 0; i < active_size; ++i) {
                int j = generator.nextInt(active_size - i);
                int temp = index[i];
                index[i] = index[j];
                index[j] = temp;
            }

            for (int s = 0; s < active_size; ++s) {

                int i = index[s];
                Example E = data.get(i);

                // G = w^T*x_i*y_i - 1
                double G = 0;
                for (int j = 0; j < E.fids.length; ++j)
                    G += param.weightvector[E.fids[j]] * E.fvals[j];
                G = G * label[i] - 1;

                // Shrinking
                if ((Math.abs(alpha[i]) < EPS && G > PGmax_old)
                        || (Math.abs(alpha[i] - C) < EPS && G < PGmin_old)) {
                    active_size--;
                    int temp = index[s];
                    index[s] = index[active_size];
                    index[active_size] = temp;
                    s--;
                    continue;
                }

                double PG = G;
                if (Math.abs(alpha[i]) < EPS) {
                    PG = Math.min(G, 0);
                } else if (Math.abs(alpha[i] - C) < EPS) {
                    PG = Math.max(G, 0);
                }

                // update alpha[i] and param.weightvector
                // if PG > 0
                if (Math.abs(PG) > 1e-12) {
                    double alpha_old = alpha[i];
                    alpha[i] = Math.min(Math.max(alpha[i] - G / QD[i], 0.0), C);
                    double d = (alpha[i] - alpha_old) * label[i];
                    for (int j = 0; j < E.fids.length; ++j)
                        param.weightvector[E.fids[j]] += d * E.fvals[j];
                }

                PGmax_new = Math.max(PGmax_new, PG);
                PGmin_new = Math.min(PGmin_new, PG);
            }// inner loop ends

            iter++;

            if (iter % 10 == 0)
                System.out.print("..");

            if (PGmax_new - PGmin_new <= eps) {
                if (active_size == N)
                    break;

                // since PGmax_old & PGmin_old are set to extremes
                // no shrinking will happen in the next iteration
                active_size = N;
                System.out.print("*");
                PGmax_old = Double.POSITIVE_INFINITY;
                PGmin_old = Double.NEGATIVE_INFINITY;
                continue;
            }

            PGmax_old = PGmax_new > 0 ? PGmax_new : Double.POSITIVE_INFINITY;
            PGmin_old = PGmin_new < 0 ? PGmin_new : Double.NEGATIVE_INFINITY;

            // System.out.println("i="+iter+", obj="+calculateObjective(param,data,C));
        }

        System.out.println("\n Done ");

        // F = sum_i(alpha[i]) + 1/2*norm(w)
        // what is this value for?
        double F = 0;
        for (int i = 0; i < param.weightvector.length; ++i)
            F += param.weightvector[i] * param.weightvector[i];
        F = F / 2;
        for (int i = 0; i < N; ++i)
            F += -alpha[i];
        F = -F;
        return F;
    }

    private double calculateObjective(WeightParameter param,
            Vector<Example> data, double C) {
        double norm = param.norm();
        double loss = 0;
        for (Example e : data) {
            loss += param.hingeLoss(e);
        }
        double objective = norm / 2 + C * loss;
        return objective;

    }

}
