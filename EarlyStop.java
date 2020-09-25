package queen;

import java.util.ArrayList;
import java.util.List;
import java.util.Stack;

public class EarlyStop {

    private Graph G;
    private int requiredVertexDegree;

    public EarlyStop(Graph G, int requiredVertexDegree) {
        this.G = G;
        this.requiredVertexDegree = requiredVertexDegree;
    }

    private boolean containsCompleteSubGraph(Graph G) {
        Stack<Integer> stack = new Stack<>();
        for (int v = 0; v < G.V(); v++) {
            if (G.degree(v) < requiredVertexDegree) {
                stack.push(v);
            }
        }
        while (!stack.empty()) {
            int v = stack.pop();
            List<Integer> neighbors = new ArrayList<>(G.adj(v));
            for (int neighbor : neighbors) {
                G.removeEdge(v, neighbor);
                if (G.degree(neighbor) < requiredVertexDegree) {
                    stack.push(neighbor);
                }
            }
        }
        for (int i = 0; i < G.V(); i++) {
            if (G.degree(i) != 0) {
                return true;
            }
        }
        return false;
    }

    public static void main(String[] args) {
        int numberOfVertices = 7;
        int numberOfLevels = 3;
        Graph G = new Graph(numberOfVertices);
        G.addEdge(0, 3);
        G.addEdge(0, 6);
        G.addEdge(1, 2);
        G.addEdge(1, 6);
        G.addEdge(2, 6);
        G.addEdge(3, 4);
        System.out.println(G);

        EarlyStop earlyStop = new EarlyStop(G, numberOfLevels - 1);
        System.out.println(earlyStop.containsCompleteSubGraph(G));
    }
}
